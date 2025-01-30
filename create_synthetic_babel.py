from os import name
from typing import Optional, List, Dict
import logging
import joblib
import numpy as np
from sinc.render.mesh_viz import visualize_meshes
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path
from sinc.utils.file_io import read_json, write_json
from sinc.data.tools.extract_pairs import extract_frame_labels
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

from sinc.tools.frank import text_to_bp, text_list_to_bp

logger = logging.getLogger(__name__)

SPLITS = ["train", "val", "test", "all", "subset"]
import spacy
import smplx
NLP_PROC = spacy.load("en_core_web_sm")
GLOB_BD_MODEL = smplx.SMPLHLayer(
            f"./data/body_models/smplh",
            model_type='smplh',
            gender='neutral',
            ext='npz'
)
def get_split(path: str, split: str, subset: Optional[str] = ''):
    assert split in SPLITS
    filepath = Path(path) / f'{split}{subset}.pth.tar'
    split_data = joblib.load(filepath)
    return split_data


def get_babel_keys(path: str):
    filepath = Path(path) / f'../babel_v2.1/id2fname/amass-path2babel.json'
    amass2babel = read_json(filepath)
    return amass2babel

def remove_turn_plus_actions(strings):
    """
    Removes any string that contains 'turn' plus 
    at least one of: 'run', 'climb', 'jog', 'walk', 'go', or 'step'.
    Returns (filtered_list, removed_indices).
    Matching is case-insensitive.
    """
    actions = {"run", "climb", "jog", "walk", "go", "step"}
    
    filtered = []
    removed_indices = []
    
    for i, text in enumerate(strings):
        text_lower = text.lower()
        if "turn" in text_lower and any(action in text_lower for action in actions):
            # This element is removed
            removed_indices.append(i)
        else:
            # Keep this element
            filtered.append(text)
    
    return filtered, removed_indices

def sublinear_downsample_in_order(data, alpha=0.5, threshold=5):
    from collections import Counter
    import math

    """
    Downsample the data while preserving the original order of items:
      1. Count frequency of each item.
      2. Decide how many of each item to keep:
         - If freq <= threshold => keep them all (no downsampling for rare items).
         - If freq > threshold  => keep_count = floor(freq^alpha) (downsampling).
      3. Create a new data list by iterating in the original order:
         - For each item, if we haven't reached the keep_count for that item, we keep it.
         - Otherwise, we remove it.
    Returns (downsampled_data, removed_indices).
    
    alpha < 1 => bigger downsampling of high-frequency items.
    threshold => below or equal to this, items are kept fully.
    """
    # 1. Count how many times each item appears
    freq_counter = Counter(data)

    # 2. Determine how many of each item we want to keep
    keep_counts = {}
    for item, freq in freq_counter.items():
        if freq <= threshold:
            keep_counts[item] = freq  # keep them all
        else:
            keep_counts[item] = max(1, int(freq**alpha))

    # 3. Build the new dataset in the original order
    used_so_far = Counter()  # track how many of each item we've kept
    downsampled_data = []
    removed_indices = []

    for idx, item in enumerate(data):
        # If we haven't reached the keep_count for this item, we keep it
        if used_so_far[item] < keep_counts[item]:
            downsampled_data.append(item)
            used_so_far[item] += 1
        else:
            # Otherwise, we remove it
            removed_indices.append(idx)

    return downsampled_data, removed_indices

def load_gpt_labels(path: str):
    if 'gpt/gpt3-labels-list.json' not in str(path):
        path = path.replace('gpt/gpt3-labels.json','gpt/gpt3-labels-list.json')
    gpt_labels = read_json(path)
    return gpt_labels

def collate_rots_and_text(lst_elements: List) -> Dict:
    # collate_datastruct = lst_elements[0]["datastruct"].transforms.collate
    batch = {# Collate with padding for the datastruct
             "rots": 
             collate_tensor_with_padding([el["rots"] for el in lst_elements]),
             "trans": 
             collate_tensor_with_padding([el["trans"] for el in lst_elements]),
             "rots_a": 
             collate_tensor_with_padding([el["rots_a"] for el in lst_elements]),
             "rots_b": 
             collate_tensor_with_padding([el["rots_b"] for el in lst_elements]),
             "trans_a": 
             collate_tensor_with_padding([el["trans_a"] for el in lst_elements]),
             "trans_b": 
             collate_tensor_with_padding([el["trans_b"] for el in lst_elements]),

             # Collate normally for the length
            #  "datastruct_a": [x["datastruct_a"] for x in lst_elements],
            #  "datastruct_b": [x["datastruct_b"] for x in lst_elements],
             "length": [x["length"] for x in lst_elements],
             # Collate the text
             "text": [x["text"] for x in lst_elements]}

    # add keyid for example
    otherkeys = [x for x in lst_elements[0].keys() if x not in batch]
    for key in otherkeys:
        batch[key] = [x[key] for x in lst_elements]

    return batch

class BABEL(Dataset):
    dataname = "BABEL"

    def __init__(self,
                 datapath: str,
                 split: str = "train",
                 sampler=None,
                 progress_bar: bool = True,
                 load_with_rot=True,
                 downsample=True,
                 tiny: bool = False,
                 synthetic: Optional[bool] = True,
                 proportion_synthetic: float = 1.0,
                 centered_compositions: Optional[bool] = False,
                 dtype: str = 'seg+seq',
                 mode: str = 'train',
                 simultaneous_max: int = 2,
                 **kwargs):
        self.simultaneous_max = simultaneous_max
        self.split = split
        self.load_with_rot = load_with_rot
        self.downsample = downsample
        self.dtype = dtype  # seg or seq or empty string for segments or sequences
        # or all of the stuff --> 'seg', 'seq', 'pairs', 'pairs_only', ''
        self.synthetic = synthetic
        
        self.proportion_synthetic = proportion_synthetic
        self.centered_compositions = centered_compositions
 
        if self.split not in SPLITS:
            raise ValueError(f"{self.split} is not a valid split")

        super().__init__()
        if tiny:
            data_for_split = get_split(path=datapath,
                                       split=split,
                                       subset='_tiny')
            self.babel_annots = read_json(
                Path(datapath) / f'../babel_v2.1/{split}.json')
        else:
            data_for_split = get_split(path=datapath, split=split)
            self.babel_annots = read_json(
                Path(datapath) / f'../babel_v2.1/{split}.json')
        gpt_path = 'deps/gpt/gpt3-labels-list.json'

        gpt_labels_full_sent = load_gpt_labels(gpt_path)

        fname2key = get_babel_keys(path=datapath)
        motion_data_rots = {}
        motion_data_trans = {}

        joints_data = {}
        texts_data = {}
        durations = {}
        dtypes = {}
        gpt_labels = {}
        if progress_bar:
            enumerator = enumerate(
                tqdm(data_for_split, f"Loading BABEL {split}"))
        else:
            enumerator = enumerate(data_for_split)

        if tiny:
            maxdata = 1000
        else:
            maxdata = np.inf

        datapath = Path(datapath)

        valid_data_len = 0
        all_data_len = 0
        # discard = read_json('/home/nathanasiou/Desktop/conditional_action_gen/teach/data/amass/amass_cleanup_seqs/BMLrub.json')
        req_dtypes = self.dtype.split('+')
        logger.info(f'The required datatypes are : {req_dtypes}')

        # CLEANED LABELS
        file_path = './clean_labels.txt'

        # Initialize an empty list to hold the action labels
        action_labels_clean = []

        # Open the file and read the contents
        with open(file_path, 'r') as file:
            for line in file:
                # Strip newline characters and any leading/trailing whitespace from each line
                action_label = line.strip()
                # Add the cleaned-up line to your list
                action_labels_clean.append(action_label)

        for i, sample in enumerator:
            # if sample['fname'] in discard:
            #     num_bad_bml += 1
            #     continue

            # from temos.data.sampling import subsample
            all_data_len += 1
            # smpl_data = {x: smpl_data[x] for x in smpl_data.files}
            nframes_total = len(sample['poses'])
            last_framerate = sample['fps']
            babel_id = sample['babel_id']
            frames = np.arange(nframes_total)
            all_actions = extract_frame_labels(self.babel_annots[babel_id],
                                               fps=last_framerate,
                                               seqlen=nframes_total,
                                               max_simultaneous=self.simultaneous_max
                                               )
            # if not valid:
            #     invalid += 1
            #     continue

            possible_actions = {
                k: v
                for k, v in all_actions.items() if k in req_dtypes
            }

            for dtype, extracted_actions in possible_actions.items():

                for index, seg in enumerate(extracted_actions):
                    if isinstance(seg[-1], str) and seg[-1] == '':
                        # 1 bad label
                        continue
                    if dtype == 'separate_pairs':
                        frames = np.arange(seg[0][0], seg[-1][1])
                        duration = [(e - s + 1) for s, e in seg]
                        duration[-1] -= 1
                        if len(duration) == 2: duration.insert(1, 0)
                    if dtype == 'spatial_pairs':
                        frames = np.arange(seg[0], seg[1])
                        duration = seg[1] - seg[0]
                    else:
                        frames = np.arange(seg[0], seg[1])
                        duration = len(frames)
                    if duration < 50 or duration > 180:
                        continue
                    # if duration < 100:
                    #     continue

                    if seg[-1] not in action_labels_clean:
                        continue
                    keyid = f'{dtype}-{babel_id}-{index}'
                    if self.dtype == 'separate_pairs':
                        texts_data[keyid] = seg[-1]
                        durations[keyid] = duration
                        # assert features.shape[0] == sum(duration), f' \nMismatch: {babel_id}, \n {seg_ids} \n {seg} \n {frames} \n {fpair} \n {duration}--> {features.shape[0]}  {sum(duration)}'
                    else:
                        if isinstance(seg[-1], tuple):
                            texts_data[keyid] = seg[-1]
                        else:
                            texts_data[keyid] = (
                                seg[-1], )
                        durations[keyid] = duration

                    gpt_labels[keyid] = []
                    if keyid == 'seq-3749-0':
                        print('asdasdasd!!')
                    try:
                        for a_t in texts_data[keyid]:
                            bp_list = text_list_to_bp(a_t, gpt_labels_full_sent)
                            # bp_list = text_to_bp(a_t, gpt_labels_full_sent)
                            gpt_labels[keyid].append(bp_list)
                    except:
                        import ipdb; ipdb.set_trace()

                    smpl_data = {
                        "poses":
                        torch.from_numpy(sample['poses'][frames]).float(),
                        "trans":
                        torch.from_numpy(sample['trans'][frames]).float()
                    }
                    # pose: [T, 22, 3, 3]

                    valid_data_len += 1
                    from sinc.data.tools.smpl import smpl_data_to_matrix_and_trans
                    smpl_data = smpl_data_to_matrix_and_trans(smpl_data,
                                                              nohands=True)
                    
                    motion_data_rots[keyid] = smpl_data.rots
                    motion_data_trans[keyid] = smpl_data.trans
 
                    dtypes[keyid] = dtype
        # motion_data_rots = {k:v for k,v in motion_data_rots.items() if v.shape[0] > 60}
        # filtered_keys_list = list(motion_data_rots.keys())

        stand_variants = ['stand up', 'stand in place',
                          'stand inplace',
                          'stand at rest', 'stand still',
                          'stand in t-pose', 't-pose', 'tpose',
                          'stand pose', 'stand forward', 'stand with chin up',
                          'look up', 'stand with arms down', 'stand at attention',
                          'stand with arms by sides', 'stand like a lad',
                          'stand in place while jump', 'just stand',
                          'stand upright', 'stand wait for some one',
                          'stand with his hands in front of he near the chest',
                          'stand hands cup in front',
                          'stand right',
                          'stand backward',
                          'stand in fight stance with lead with left foot',
                          'stand in front of table']
        walk_variants = ['walk in place', 'walk forward hold object',
                         'walk like monkey',
                         'just walk around', 'walk and trip',
                         'front face square pattern walk',
                         'single step walk', 'jog in place',
                         'jog forward hold object',
                         'chicken walk', 'walk fast in place',
                         'walk slow in place', 'run in place']
        look_vars = ['look at wrist', 'sort things', 'look straight', 'look down', 'look up']
        meaningless = ['put up guard','push back left', 'back to position',
                       'look side to side', 'look through portal', 
                       'lower and swing body',
                       'crouch with hold at chest level in front of they', 
                       'look forward', 'chase insect', 'lie stretch',
                       'lie on the board', 'step back face off camera',
                       'elephant trunk raise', 'take phone', 
                       'look at right wrist', 'acquire and set up ball',
                       'look into distance', 'look to the ground',
                       'look at ground']
        impossible = ['jump rope', 'pretend to jog', 'jump high', 'turn body',
                      'frog jump', 'run and turn', 'jump in circles',
                      'full back bridge',
                      'half back bridge', 'jump high', 'sit and stand',
                      'turn jump', 'just jump', 'swim in place', 'run motion',
                      'rotate jump', 'jump ropes', 'sort through items',
                      'hand up high jump','look back from right', 'jump and spin',
                      'jump rope criss cross', 
                      'shift weight', 'walk forward feet brace and push',
                      'stand with feet shoulder width apart',
                      'jump rope with alternate foot steps', 
                      'alternate hands hold rail',
                      'jump rope with both feet same time','step over jump rope',
                      'take give', 'take back','drop it back', 
                      'return object', 'put things back', 'sort things on air',
                      'jog with object', 'march in place', 'ice skate','run in slow mo',
                      'hands hold hands', 
                      'forward knee bend right',  'swing body',
                      'jump spin', 'jump off', 'jump rope both feet same time',
                      'jump around', 'ice skate jump',
                      'straight jump with full twist', 'jump spin',
                      'jump jack', 'inverse jump jack', 'jump jacks',
                      'untangle self from jump rope',
                      'jump over an obstacle', 'elephant trunk swing', 
                      'turn crank', 'change it', 'bend toe touch',
                      'step off', 'look at the floor',
                      'duck walk', 'toe touch',
                      'back flip', 'hold nose',
                      'look at clock', 'look around nervously', 'touch eye',
                      'check time', 'sit and write', 
                      'turn walk back while bend elbows inward',
                      'turn dial', 'turn full', 'bend knees then stand',
                      'stand with hands halfway out','put toothpaste on brush with right hand',
                      'open the locker', 'walk front and turn walk back'
                      ]

        texts_data ={k:v for k,v in texts_data.items() if v[0] not in stand_variants}
        texts_data ={k:v for k,v in texts_data.items() if v[0] not in walk_variants}
        texts_data ={k:v for k,v in texts_data.items() if v[0] not in impossible}
        texts_data ={k:v for k,v in texts_data.items() if v[0] not in meaningless}
        texts_data ={k:v for k,v in texts_data.items() if v[0] not in look_vars}
        texts_data ={k:v for k,v in texts_data.items() if 'while' not in v[0]}
        texts_data ={k:v for k,v in texts_data.items() if 'look up' not in v[0]}
        texts_data ={k:v for k,v in texts_data.items() if 'posture' not in v[0]}
        texts_data ={k:v for k,v in texts_data.items() if 'touch screen' not in v[0]}

        # dif_walks = []
        # for k, v in texts_data.items():
        #     if 'walk' in v[0]:
        #         dif_walks.append(v[0])

        # Keep only the long walks
        short_walks = []
        for k, v in texts_data.items():
            if 'walk' in v[0]:
                if durations[k] < 90:
                    short_walks.append(k)
        texts_data = {k:v for k,v in texts_data.items() if k not in short_walks}

        # Keep only some of the walks
        different_walks = []
        from collections import Counter
        for k, v in texts_data.items():
            if 'walk' in v[0]:
                different_walks.append(v[0])
        freqs = dict(Counter(different_walks))
        mock_text = {}
        for k, v in texts_data.items():
            mock_text[k] = v[0]

        inverse_dict = {}
        for key, value in mock_text.items():
            inverse_dict.setdefault(value, []).append(key)
        import itertools
        #import ipdb;ipdb.set_trace()
        for k, v in texts_data.items():
            if v[0] in freqs and freqs[v[0]] > 2:
                # import ipdb;ipdb.set_trace()
                maxi = durations[ inverse_dict[v[0]][0] ]
                maxkey = inverse_dict[v[0]][0]
                for subkey in inverse_dict[v[0]]:
                    if durations[subkey] > maxi and durations[subkey] < 200:
                        maxi = durations[subkey]
                        maxkey = subkey
                inverse_dict[v[0]] = [maxkey]
        keys_to_keep = list(itertools.chain(*list(inverse_dict.values())))
        #import ipdb;ipdb.set_trace()
        texts_data = {k:v for k,v in texts_data.items() if k in keys_to_keep}
        motion_data_rots = {k:v for k,v in motion_data_rots.items() if k in keys_to_keep}
        motion_data_trans = {k:v for k,v in motion_data_trans.items() if k in keys_to_keep}
        gpt_labels = {k:v for k,v in gpt_labels.items() if k in keys_to_keep}
        # import ipdb;ipdb.set_trace()

        # r = get_offscreen_renderer('./data/motion-editing-project/body_models')
        # smpl_params = pack_to_render(trans=trans_comb, rots=rots_comb, pose_repr='rot')
        # render_motion(r, smpl_params, pose_repr='aa', filename=f"./xvid3")
        ####################UNKOWN####################
        # filtering very common actions
        # different_texts = []
        # for k, v in texts_data.items():
        #     different_texts.append(v[0])
        # freqs = dict(Counter(different_texts))
        # mock_text = {}
        # for k, v in texts_data.items():
        #     mock_text[k] = v[0]

        # inverse_dict = {}
        # for key, value in mock_text.items():
        #     inverse_dict.setdefault(value, []).append(key)
        # import itertools

        # for k, v in texts_data.items():
        #     if v[0] in freqs and freqs[v[0]] > 1:
        #         maxi = durations[ inverse_dict[v[0]][0] ]
        #         maxkey = inverse_dict[v[0]][0]
        #         for subkey in inverse_dict[v[0]]:
        #             if durations[subkey] > maxi and durations[subkey] < 200:
        #                 maxi = durations[subkey]
        #                 maxkey = subkey
        #         inverse_dict[v[0]] = [maxkey]
        # keys_to_keep = list(itertools.chain(*list(inverse_dict.values())))
        # texts_data = {k:v for k,v in texts_data.items() if k in keys_to_keep}
        # motion_data_rots = {k:v for k,v in motion_data_rots.items() if k in keys_to_keep}
        # motion_data_trans = {k:v for k,v in motion_data_trans.items() if k in keys_to_keep}
        # gpt_labels = {k:v for k,v in gpt_labels.items() if k in keys_to_keep}
        #################### UNKNOWN ####################

        ###### synth_seg-1644-0+seg-3276-0
        if synthetic:
            # from a keyid, prepare what keyids is possible to be chosen
            from sinc.info.joints import get_compat_matrix
            self.compat_matrix = get_compat_matrix(gpt_labels_full_sent)

            from collections import defaultdict
            self.keyids_from_text = defaultdict(list)

            self.compat_seqs = {}
            for key, val in texts_data.items():
                if "seq" not in key and "seg" not in key:
                    continue
                self.keyids_from_text[val[0]].append(key)

            for key, val in texts_data.items():
                self.compat_seqs[key] = [
                    y for x in self.compat_matrix[val[0]]
                    for y in self.keyids_from_text[x]
                ]
        all_compat_tuples = [(k, v) for k, values in self.compat_seqs.items() for v in values]
        sorted_tuples = [tuple(sorted(t)) for t in all_compat_tuples]
        sorted_tuples_wo_dups = list(set(sorted_tuples))
        # flattened_list = list(itertools.chain(*sorted_tuples_wo_dups))
        # for k, v 

        ########################################################################
        all_texts_comb = []
        for k in sorted_tuples_wo_dups:
            all_texts_comb.append( (
                                texts_data[k[0]][0],
                                texts_data[k[1]][0]
                                )
                                )
        #import ipdb;ipdb.set_trace()
        texts_combs = rule_based_concat(all_texts_comb, 
                                        conj_word='while')
        texts_combs = [ txt_c[0] for txt_c in texts_combs ]
        idxs_too_short = []
        for idx, el in enumerate(sorted_tuples_wo_dups):
            if min(motion_data_rots[el[0]].shape[0], motion_data_rots[el[1]].shape[0]) < 75:
                idxs_too_short.append(idx)
        texts_combs = [text for i, text in enumerate(texts_combs) if i not in idxs_too_short]
        #import ipdb;ipdb.set_trace()
        sorted_tuples_wo_dups = [k for i, k in enumerate(sorted_tuples_wo_dups) if i not in idxs_too_short]
        # texts_combs_freqs = Counter(texts_combs)
        final_texts_subs, idxs_to_remove = sublinear_downsample_in_order(texts_combs, 
                                                                    alpha = 0.5, 
                                                                    threshold=10)
        sorted_tuples_wo_dups = [k for i, k in enumerate(sorted_tuples_wo_dups) if i not in idxs_to_remove]
        #import ipdb;ipdb.set_trace()
        final_texts, idxs_to_remove_incompat = remove_turn_plus_actions(final_texts_subs)
        sorted_tuples_wo_dups = [k for i, k in enumerate(sorted_tuples_wo_dups) if i not in idxs_to_remove_incompat]
        filter_similars = False
        indices_to_remove = set()

        if filter_similars:
            from sentence_transformers import SentenceTransformer, util
            model = SentenceTransformer("all-MiniLM-L6-v2")
            # Two lists of sentences
            # Compute embedding for both lists
            # for k in sorted_tuples_wo_dups:
            #     text_comb_rand_conj = rule_based_concat([ (
            #                          texts_data[k[0]][0],
            #                          texts_data[k[1]][0]
            #                          ) 
            #                       ])
            #     texts_combs.append(f'{text_comb_rand_conj[0][0]}')
            # texts_combs = list(set(texts_combs))
            
            ### text combinations
            
            embeddings1 = model.encode(texts_combs, convert_to_tensor=True)
            cosine_scores = util.cos_sim(embeddings1, embeddings1)
            cosine_scores = cosine_scores.triu()

            threshold = 0.98
            k = 1  # Number of most similar texts to remove
            indices_to_remove = set()

            # Get the indices of the k most similar texts for each text
            for i in range(cosine_scores.shape[0]-1):
                _, indices = cosine_scores[i].topk(k+1)  # +1 because the most similar text to a text is the text itself
                ind_over_threshold = []
                for ii in indices[1:].tolist():
                    if cosine_scores[i, ii] > threshold:
                        ind_over_threshold.append(ii)
                if ind_over_threshold:
                    indices_to_remove.update(ind_over_threshold)  # Exclude the first index because it's the index of the text itself
                    # print(len(ind_over_threshold))
                    # print(f'{texts_combs[i]} is similar to: {texts_combs[ind_over_threshold[0]]} ==> {cosine_scores[i, ind_over_threshold[0]]}')

        # Create a new list that includes only the texts not in indices_to_remove
        texts_combs = [text for i, text in enumerate(final_texts) if i not in indices_to_remove]
        sorted_tuples_wo_dups = [k for i, k in enumerate(sorted_tuples_wo_dups) if i not in indices_to_remove]

        ########################################################################
        # embeddings2 = model.encode(xx, convert_to_tensor=True)
        # Compute cosine-similarities
        rots1 = []
        rots2 = []
        trans1 = []
        trans2 = []
        texts = []
        gpt_lbs_FINAL = []
        # 8985 | 30729
        keyids_comb = []
        for el in tqdm(sorted_tuples_wo_dups):
            # if el[0] == "seq-6558-0" or el[1] == "seq-6558-0":
            rots1.append(motion_data_rots[el[0]])
            rots2.append(motion_data_rots[el[1]])
            trans1.append(motion_data_trans[el[0]])
            trans2.append(motion_data_trans[el[1]])
            if el[0] == 'seg-11577-5' and el[1] == 'seq-6041-0':
                print('buhfasd1')
            if el[0] == 'seg-155-3' and el[1] == 'seq-3749-0':
                print('buhfasd1222222')
            # from body_renderer import get_render
            # get_render(GLOB_BD_MODEL, 
            #         smpl_data.trans, 
            #         smpl_data.rots[:, 0],
            #         smpl_data.rots[:, 1:],
            #         output_path='./output_movie.mp4',
            #             text='',
            #             colors='sky blue'
            #         )
            gpt_lbs_FINAL.append([gpt_labels[el[0]][0], gpt_labels[el[1]][0]])
            keyids_comb.append(f'{el[0]}+{el[1]}')

        # import ipdb; ipdb.set_trace()

        comb_rots, comb_trans, _ = transform_batch_to_mixed_synthetic_ds(
                                                                    rots_1=rots1, 
                                                                    rots_2=rots2,
                                                                    trans_1=trans1,
                                                                    trans_2=trans2,
                                                                    gpt_labels_batch=gpt_lbs_FINAL,
                                                                    keyids_to_check=keyids_comb)
        keyids1, keyids2 = zip(*sorted_tuples_wo_dups)

        rots1 = dict(zip(keyids1, rots1))
        trans1 = dict(zip(keyids1, trans1))
        rots2 = dict(zip(keyids2, rots2))
        trans2 = dict(zip(keyids2, trans2))
        comb_rots = dict(zip(keyids_comb, comb_rots))
        comb_trans = dict(zip(keyids_comb, comb_trans))
        texts_data = dict(zip(keyids_comb, texts_combs))

        single_rots = rots1.copy()
        single_rots.update(rots2)
        single_trans = trans1.copy()
        single_trans.update(trans2)

        # motion_data_trans = {} 
        # motion_data_rots = {}
        # for idx in range(len(keyids_comb)):
        #     motion_data_rots[keyids_comb[idx]] = motion_data_rots[idx]
        #     motion_data_trans[keyids_comb[idx]] = motion_data_trans[idx]
        #     texts_data[keyids_comb[idx]] = (f'{texts_data[sorted_tuples_wo_dups[idx][0]][0]} while {texts_data[sorted_tuples_wo_dups[idx][1]][0]}',)
        self.single_rots = single_rots
        self.single_trans = single_trans
        self.motion_data_trans = comb_trans
        self.motion_data_rots = comb_rots

        self.texts_data = texts_data
        self._split_index = list(self.motion_data_rots.keys())
        self._num_frames_in_sequence = durations
        self.keyids = list(self.motion_data_rots.keys())
        self.gpt_labels = gpt_labels

    def _load_text(self, keyid):
        sequences = self.texts_data[keyid]
        return sequences

    def load_keyid(self, keyid, mode='train', proportion=None):
        text = self._load_text(keyid)

        # check_compat = self.synthetic and self.sample_dtype[keyid] in ['seg', 'seq'] and (self.compat_seqs[keyid])
        # GPT CASE
        # check compatibility with synthetic
        # randomly do synthetic data or not
        # if there is some compatibility
        # take a random pair
        keyida, keyidb = keyid.split('+')

        rots_comb = self.motion_data_rots[keyid]
        trans_comb = self.motion_data_trans[keyid]

        rots_a = self.single_rots[keyida]
        rots_b = self.single_rots[keyidb]
        trans_a = self.single_trans[keyida]
        trans_b = self.single_trans[keyidb]

        length = min(len(rots_a), len(rots_b))

        # force lenght constraint to be centered
        start_1 = (len(rots_a) - length)//2
        rots_a = rots_a[start_1:start_1+length]
        trans_a = trans_a[start_1:start_1+length]

        start_2 = (len(rots_b) - length)//2
        rots_b = rots_b[start_2:start_2+length]
        trans_b = trans_b[start_2:start_2+length]

        element = { 'rots': rots_comb,
                    'trans': trans_comb,
                    'length': [len(rots_a), len(rots_b)], 
                    'text': text,
                    'keyid': f"synth_{keyid}",
                    'rots_a': rots_a,
                    'trans_a': trans_a,
                    'rots_b': rots_b,
                    'trans_b': trans_b,
                  }
        return element


    def __getitem__(self, index):
        keyid = self._split_index[index]
        return self.load_keyid(keyid, mode='train')

    def __len__(self):
        return len(self._split_index)

    def __repr__(self):
        return f"{self.dataname} dataset: ({len(self)}, _, ..)"



def gpt_compose(rots1, trans1, rots2, trans2, bp1, bp2, 
                center=True, squeeze=False):
    import torch
    from sinc.info.joints import smpl2gpt, gpt_side, smpl_bps
    from sinc.info.joints import smpl_bps_ids_list, smpl_bps_ids, smplh_joints
    from sinc.info.joints import smpl_bps_ids

    # minimum bp should be 2
    if sum(bp1) < sum(bp2):
        bp1, bp2 = bp2, bp1
        rots1, rots2 = rots2, rots1
        trans1, trans2 = trans2, trans1

    if squeeze:
        rots1, trans1 = torch.squeeze(rots1), torch.squeeze(trans1)
        rots2, trans2 = torch.squeeze(rots2), torch.squeeze(trans2)

    # STEP 1: same length with centering
    # common length
    length_1 = len(rots1)
    length_2 = len(rots2)

    # assumption 1
    # should be the same lenght
    length = min(length_1, length_2)

    if center:
        # force lenght constraint to be centered
        start_1 = (length_1 - length)//2
        rots1 = rots1[start_1:start_1+length]
        trans1 = trans1[start_1:start_1+length]

        start_2 = (length_2 - length)//2
        rots2 = rots2[start_2:start_2+length]
        trans2 = trans2[start_2:start_2+length]
    else:
        # trim length
        rots1 = rots1[:length]
        trans1 = trans1[:length]
        rots2 = rots2[:length]
        trans2 = trans2[:length]

    # assumption 2:
    # For composition, the two legs + global should be packed together
    left_leg_id = smpl_bps_ids_list.index("left leg")
    right_leg_id = smpl_bps_ids_list.index("right leg")
    global_id = smpl_bps_ids_list.index("global")

    if bp2[left_leg_id] or bp2[right_leg_id] or bp2[global_id]:
        bp1[left_leg_id] = 0
        bp1[right_leg_id] = 0
        bp1[global_id] = 0
        bp2[left_leg_id] = 1
        bp2[right_leg_id] = 1
        bp2[global_id] = 1
    else:
        bp1[left_leg_id] = 1
        bp1[right_leg_id] = 1
        bp1[global_id] = 1
        bp2[left_leg_id] = 0
        bp2[right_leg_id] = 0
        bp2[global_id] = 0
    # bp2 is minimum, will be added at the end (override)
    # add more to bp1

    # assumption 3:
    # binary selection of everything
    for i, x2 in enumerate(bp2):
        if x2 == 0:
            bp1[i] = 1

    body_parts_1 = [smpl_bps_ids_list[i] for i, x in enumerate(bp1) if x == 1]
    body_parts_1 = [y for x in body_parts_1 for y in smpl_bps[x]]

    body_parts_2 = [smpl_bps_ids_list[i] for i, x in enumerate(bp2) if x == 1]
    body_parts_2 = [y for x in body_parts_2 for y in smpl_bps[x]]

    # STEP 2: extract the body_parts
    joints_1 = [smplh_joints.index(x) for x in body_parts_1]
    joints_2 = [smplh_joints.index(x) for x in body_parts_2]

    frank_rots = torch.zeros_like(rots1)
    frank_rots[:, joints_1] = rots1[:, joints_1]
    frank_rots[:, joints_2] = rots2[:, joints_2]

    # assumption 3
    # gravity base hand crafted translation rule
    if "foot" in " ".join(body_parts_1):
        frank_trans = trans1
    else:
        frank_trans = trans2

    return frank_rots, frank_trans


def transform_batch_to_mixed_synthetic_ds(rots_1, rots_2, trans_1, trans_2,
                                          gpt_labels_batch, keyids_to_check):
    rots_lst = []
    trans_lst = []
    lens_lst = []
    assert len(rots_1) == len(rots_2)
    nofels = len(rots_1)
    # bugseq1 = (key_a == 'seg-11577-5' and key_b == 'seq-6041-0')
    # bugseq2 = (key_a == 'seg-155-3' and key_b == 'seq-3749-0')

    # import ipdb; ipdb.set_trace()
    for idx in range(nofels):
        lena = len(rots_1[idx])
        lenb = len(rots_2[idx])
        rots_a = rots_1[idx]
        rots_b = rots_2[idx]
        trans_a = trans_1[idx]
        trans_b = trans_2[idx]
        keya, keyb = keyids_to_check[idx].split('+')
        if keya == 'seg-11577-5' and keyb == 'seq-6041-0':
            print('buhfasd1')
        if keya == 'seg-155-3' and keyb == 'seq-3749-0':
            print('buhfasd1222222')
        if keya == 'seq-8048-0' and keyb == 'seq-8500-0':
            print('buhfasd1222222')
        if keya == 'seq-3749-0' and keyb == 'seq-8048-0':
            print('buhfasd1222222')
        bp1_copy = gpt_labels_batch[idx][0].copy()
        bp2_copy = gpt_labels_batch[idx][1].copy()

        # from body_renderer import get_render
        # get_render(GLOB_BD_MODEL, 
        #            trans_a, 
        #            transform_body_pose(rots_a, 'rot->aa')[:, 0],
        #            transform_body_pose(rots_a, 'rot->aa')[:, 1:],
        #            output_path='./output_movie.mp4',
        #             colors=['sky blue']
        #            )
        # get_render(
        #     GLOB_BD_MODEL,
        #     [edited_motion_transformed['trans'].detach().cpu(),
        #      source_motion_transformed['trans'].detach().cpu()],
        #     [edited_motion_transformed['rots_init'].detach().cpu(),
        #      source_motion_transformed['rots_init'].detach().cpu()],
        #     [edited_motion_transformed['rots_rest'].detach().cpu(),
        #      source_motion_transformed['rots_rest'].detach().cpu()],
        #     output_path='./output_movie.mp4',
        #     text='',
        #     colors=['sky blue', 'red']
        # )
        rots_comb, trans_comb = gpt_compose(rots_a, trans_a,
                                            rots_b, trans_b, 
                                            bp1_copy,
                                            bp2_copy,
                                            center=True)
        
        curlen = len(rots_comb) 
        rots_lst.append(rots_comb)
        trans_lst.append(trans_comb)
        lens_lst.append(curlen)
    return rots_lst, trans_lst, lens_lst


def gerund_augment(text_list):
    text_dur_gerund = []
    for x in text_list:
        occ = 0
        sample_ger = []

        for wd in NLP_PROC(x):
            if wd.pos_ == 'VERB':
                occ += 1
            if occ == 2 and wd.pos_ == 'VERB':
                if wd.text.endswith('ing'):
                    sample_ger.append(wd.text)
                else:
                    sample_ger.append(f'{wd.text}ing') 
            else:
                sample_ger.append(wd.text) 

        sample_ger = ' '.join(sample_ger)
        text_dur_gerund.append(sample_ger)
    return text_dur_gerund

def rule_based_concat(texts, conj_word=None):
    conj_word_dict = {'while':0, 'sim': 1, ',': 2, 'same_time':3, 'during': 4}
    from random import randint
    texts_wl = [" while ".join(x) for x in texts]
    # texts_wl = gerund_augment(texts_wl)

    texts_sim = [(" <and> ".join(x),)  for x in texts]
    texts_sim = [ f"simultaneously {x[0]}" if '<and>' in x[0] else x[0] \
                    for x in texts_sim ]
    texts_sim = [ x.replace('<and>', 'and') for x in texts_sim ]

    texts_and_same = [ (f" <and> ".join(x),) for x in texts]
    texts_and_same = [ f"{x[0]} at the same time"\
        if '<and>' in x[0] else x[0] for x in texts_and_same ]
    texts_and_same = [ x.replace('<and>', 'and') for x in texts_and_same ]
    
    texts_dur = [ f" during ".join(x) for x in texts]
    # texts_dur = gerund_augment(texts_dur)
    
    text_aug_batch = []
    conj_word = 'while'
    for augm_text_el in zip(texts_wl, texts_sim, texts_and_same,
                            texts_dur):
        if conj_word is None:

            text_aug_batch.append((augm_text_el[randint(0, 3)],))
        else:
            augm_idx = conj_word_dict[conj_word]
            text_aug_batch.append((augm_text_el[augm_idx],))

    assert len(text_aug_batch) == len(texts)
    assert sum([len(x) for x in text_aug_batch]) == len(texts)
    return text_aug_batch


from sinc.data.tools.collate import collate_tensor_with_padding
# lens, motions_ds = self.transform_batch_to_mixed_synthetic(batch)

from sinc.tools.visuals import pack_to_render, render_motion, get_offscreen_renderer


dataset = BABEL(datapath='data/babel/babel-smplh-30fps-male',
                synthetic=True)
from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=32,
                        shuffle=False, num_workers=4,
                        collate_fn=collate_rots_and_text)
database_to_save = {}
cnt = 0


locombs = []

from einops import rearrange
def _canonica_facefront(rotations, translation):

    rots_motion = rotations
    trans_motion = translation
    datum_len = rotations.shape[0]
    from sinc.tools.transform3d import canonicalize_rotations, transform_body_pose

    rots_motion_rotmat = transform_body_pose(rots_motion.reshape(datum_len,
                                                        -1, 3),
                                                        'aa->rot')
    orient_R_can, trans_can = canonicalize_rotations(rots_motion_rotmat[:, 0],
                                                     trans_motion) 
    rots_motion_rotmat_can = rots_motion_rotmat
    rots_motion_rotmat_can[:, 0] = orient_R_can
    translation_can = trans_can - trans_can[0]
    rots_motion_aa_can = transform_body_pose(rots_motion_rotmat_can,
                                                'rot->aa')
    rots_motion_aa_can = rearrange(rots_motion_aa_can, 'F J d -> F (J d)',
                                    d=3)
    return rots_motion_aa_can, translation_can

def pipe_to_json_structured(vid_ids_a, vid_ids_b, stamps_a, stamps_b,
                            path_to_save):
    template_fp = '/fast/nathanasiou/VIDID'
    # vid_ids_a, vid_ids_b, stamps_a, stamps_b = get_vids_to_label()
    # vid_ids_a = ['10341']
    # vid_ids_b = ['10486']
    # stamps_a = [(10, 17.1)]
    # stamps_b = [(2, 3.1)]
    stamps_a = [{'begin': s, 'end': e} for (s, e) in stamps_a]
    stamps_b = [{'begin': s, 'end': e} for (s, e) in stamps_b]


    final_vids = {'motion_a': [template_fp.replace('VIDID', x) for x in vid_ids_a],
                  'motion_b': [template_fp.replace('VIDID', x) for x in vid_ids_b],
                  'stamps_a': stamps_a,
                  'stamps_b': stamps_b}
    print('The json file that must be given to the next script is stored in:', f'{path_to_save}')
    write_json(final_vids, f'{path_to_save}')

dict_to_save = {'motion_a': [], 
                'motion_b':[], 
                'stamps_a':[], 
                'stamps_b':[]
                }

def pipe_to_json_structured_triplets(vid_ids_a, 
                                     vid_ids_b, 
                                     vid_ids_c, 
                                     stamps_a, 
                                     stamps_b,
                                     stamps_c,
                                     path_to_save):
    template_fp = '/fast/nathanasiou/VIDID'
    # vid_ids_a, vid_ids_b, stamps_a, stamps_b = get_vids_to_label()
    # vid_ids_a = ['10341']
    # vid_ids_b = ['10486']
    # stamps_a = [(10, 17.1)]
    # stamps_b = [(2, 3.1)]
    stamps_a = [{'begin': s, 'end': e} for (s, e) in stamps_a]
    stamps_b = [{'begin': s, 'end': e} for (s, e) in stamps_b]
    stamps_c = [{'begin': s, 'end': e} for (s, e) in stamps_c]


    final_vids = {'motion_a': [template_fp.replace('VIDID', x) for x in vid_ids_a],
                  'motion_b': [template_fp.replace('VIDID', x) for x in vid_ids_b],
                  'motion_c': [template_fp.replace('VIDID', x) for x in vid_ids_c],
                  'stamps_a': stamps_a,
                  'stamps_b': stamps_b,
                  'stamps_c': stamps_c
                  }
    print('The json file that must be given to the next script is stored in:', f'{path_to_save}')
    write_json(final_vids, f'{path_to_save}')

dict_to_save = {'motion_a': [], 
                'motion_b':[], 
                'motion_c':[], 

                'stamps_a':[], 
                'stamps_b':[],
                'stamps_c':[]

                }

# r = get_offscreen_renderer('./data/motion-editing-project/body_models')
keys1 = []
keys2 = []
keys3 = []
vids_1 = []
vids_1_mot = []
vids_2 = []
vids_2_mot = [] 
vids_3 = []
vids_3_mot = [] 

stamps_1 = []
stamps_2 = []
stamps_3 = []

for bs in tqdm(dataloader):
    # comb_text = rule_based_concat(bs['text'])
    from sinc.tools.geometry import matrix_to_axis_angle
    text_a = []
    text_b = []

    for text_sim in bs['text']:
        t_a, t_b = text_sim.split('while')
        text_a.append(t_a.strip())
        text_b.append(t_b.strip())
    
    #for x in text_a:
    #    if 'walk' in x:
    #        import ipdb;ipdb.set_trace()
    #for x in text_b:
    #    if 'walk' in x:
    #        import ipdb;ipdb.set_trace()
    # continue
    for jj in range(len(bs['keyid'])):
        lenmo = bs['length'][jj][0]
        c1 = 'walk' in bs['text'][jj] and 'turn' in bs['text'][jj] 
        c2 = 'stroll' in bs['text'][jj] and 'turn' in bs['text'][jj] 
        c3 = 'step' in bs['text'][jj] and 'turn' in bs['text'][jj] 
        c4 = 'run' in bs['text'][jj] and 'turn' in bs['text'][jj] 
        c5 = 'jog' in bs['text'][jj] and 'turn' in bs['text'][jj] 
        cur_text_comb_while = bs['text'][jj]
        # c6 = 'jump' in cur_text_comb_while or 'incline' in cur_text_comb_while or 'climb' in cur_text_comb_while or 'hop' in cur_text_comb_while or 'spin' in cur_text_comb_while
        # c14 = 'bend down' in cur_text_comb_while and 'walk' in cur_text_comb_while
        # c15 = 'bend down' in cur_text_comb_while and 'jog' in cur_text_comb_while
        # c16 = 'bend down' in cur_text_comb_while and 'run' in cur_text_comb_while
        meet_condition = c1 or c2 or c3 or c4 or c5
        if meet_condition:
            continue
        rots_a = matrix_to_axis_angle(bs['rots_a'][jj][:lenmo])
        rots_b = matrix_to_axis_angle(bs['rots_b'][jj][:lenmo])
        rots_comb = matrix_to_axis_angle(bs['rots'][jj][:lenmo])

        trans_a = bs['trans_a'][jj][:lenmo]
        trans_b = bs['trans_b'][jj][:lenmo]
        trans_comb = bs['trans'][jj][:lenmo]
        # additions
        rots_a_can, trans_a_can = _canonica_facefront(rots_a, trans_a)
        rots_b_can, trans_b_can = _canonica_facefront(rots_b, trans_b)
        rots_comb_can, trans_comb_can = _canonica_facefront(rots_comb,
                                                            trans_comb)
        cands = ["synth_seg-1644-0+seg-3276-0",
                 "synth_seq-146-0+seq-4257-0"]
        if bs['keyid'][jj] in cands:
            x = 1
        # 
        # "synth_seq-146-0+seq-4257-0"

        # ii = 0
        # for mot in [[rots_a_can, trans_a_can],
        #             [rots_b_can, trans_b_can],
        #             [rots_comb_can, trans_comb_can]]:
        #     smpl_params = pack_to_render(trans=mot[1], 
        #                                 rots= mot[0], pose_repr='aa')
        #     render_motion(r, smpl_params, pose_repr='aa',
        #                     filename=f"./some_vid_{bs['keyid'][jj]}-{ii}")
        #     ii+=1

        keyid = f'sinc_synth_{str(cnt).zfill(6)}'
        locombs.append(bs['keyid'][jj])
        keyid_comb = bs['keyid'][jj]
        key_a, key_b = keyid_comb.split('+')
        bugseq1 = (key_a == 'synth_seg-11577-5' and key_b == 'seq-6041-0')
        bugseq2 = (key_a == 'synth_seg-155-3' and key_b == 'seq-3749-0')
        
        if bugseq1 or bugseq2:
            print('iasdasd')
        # wave and run
        
        # wave | run --> 4 combinations
        # a, comba
        # synth_seg-11577-52Bseq-6041-0_0_112.mp4
        database_to_save[f'{keyid}'] = {'motion_1': {'rots':rots_a_can,
                                                        'trans': trans_a_can,
                                                        'babel_key': key_a,
                                                        'babel_text': text_a[jj]},
                                      'motion_2': {'rots': rots_b_can,
                                                        'trans': trans_b_can,
                                                        'babel_key': key_b,
                                                        'babel_text': text_b[jj]},
                                      'motion_combo': {'rots': rots_comb_can,
                                                        'trans': trans_comb_can,
                                                        'babel_key': keyid_comb,
                                                        'babel_text': bs['text'][jj]},

                                      'text': bs['text'][jj]}
        vids_1.append(key_a)
        vids_1_mot.append({'rots':rots_a_can, 
                           'trans':trans_a_can})
        vids_2.append(key_b)
        vids_2_mot.append({'rots':rots_b_can, 
                           'trans':trans_b_can})
        vids_3.append(keyid_comb)
        vids_3_mot.append({'rots':rots_comb_can, 
                           'trans':trans_comb_can})

        stamps_1.append((0, len(rots_a_can)))
        stamps_2.append((0, len(rots_b_can)))
        stamps_3.append((0, len(rots_comb_can)))

        cnt += 1
#import ipdb;ipdb.set_trace()
if True:
    # remove dups get indices
    path_to_save = 'data/motion-editing-project/sinc_synth_babel-amass/pkls'

    # unique_dict1 = {item: index for index, item in enumerate(vids_1)}
    # indices1 = list(unique_dict1.values())
    # vids_1 = [vids_1[i] for i in indices1]
    # vids_1_mot = [vids_1_mot[i] for i in indices1]
    vids_1 = [f'{path_to_save}/{x}' for x in vids_1]

    # unique_dict2 = {item: index for index, item in enumerate(vids_2)}
    # indices2 = list(unique_dict2.values())
    # vids_2 = [vids_2[i] for i in indices2]
    # vids_2_mot = [vids_2_mot[i] for i in indices2]
    vids_2 = [f'{path_to_save}/{x}' for x in vids_2]
    vids_3 = [f'{path_to_save}/{x}' for x in vids_3]

    import os
    for j, motion_to_save in enumerate(tqdm(vids_1_mot)):
        if not os.path.isfile(f'{vids_1[j]}.pth.tar'):
            joblib.dump(motion_to_save, f'{vids_1[j]}.pth.tar')

    for j, motion_to_save in enumerate(tqdm(vids_2_mot)):
        if not os.path.isfile(f'{vids_2[j]}.pth.tar'):
            joblib.dump(motion_to_save, f'{vids_2[j]}.pth.tar')

    for j, motion_to_save in enumerate(tqdm(vids_3_mot)):
        if not os.path.isfile(f'{vids_3[j]}.pth.tar'):
            joblib.dump(motion_to_save, f'{vids_3[j]}.pth.tar')

    pipe_to_json_structured_triplets(vids_1, vids_2, vids_3, 
                            stamps_1, stamps_2,
                            stamps_3,
                            path_to_save='data/motion-editing-project/sinc_synth_babel-amass/synthetic_selected.json')
x=1
joblib.dump(database_to_save, 
            'data/motion-editing-project/sinc_synth_babel-amass/sinc_synth_edits_v3.pth.tar')
import json
print(len(database_to_save))
with open('data/motion-editing-project/sinc_synth_babel-amass/sinc_synth_keys_used.json', 'w') as f:
    json.dump(locombs, f)
