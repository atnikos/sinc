import os
import sys
from sinc.tools.transform3d import transform_body_pose
import torch
from aitviewer.headless import HeadlessRenderer

def get_offscreen_renderer(path_tobody_models='data/body_models/'):
    import os
    os.system("Xvfb :12 -screen 1 640x480x24 &")
    os.environ['DISPLAY'] = ":12"
    from aitviewer.configuration import CONFIG as AITVIEWER_CONFIG
    from aitviewer.headless import HeadlessRenderer
    AITVIEWER_CONFIG.update_conf({"playback_fps": 30,
                                    "auto_set_floor": True,
                                    "smplx_models": path_tobody_models,
                                    "z_up": True})
    return HeadlessRenderer()

def detach_to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def remove_padding(tensors, lengths):
    return [tensor[:tensor_length] for tensor, tensor_length in zip(tensors, lengths)]

def pack_to_render(rots, trans, pose_repr='6d'):
    # make axis-angle
    # global_orient = transform_body_pose(rots, f"{pose_repr}->aa")
    if pose_repr != 'aa':
        body_pose = transform_body_pose(rots, f"{pose_repr}->aa")
    else:
        body_pose = rots

    if body_pose.shape[-1] < 9:
        body_pose = body_pose.flatten(1)

    if trans is None:
        trans = torch.zeros((rots.shape[0], rots.shape[1], 3),
                             device=rots.device)
    render_d = {'body_transl': trans,
                'body_orient': body_pose[..., :3],
                'body_pose': body_pose[..., 3:]}
    return render_d


def render_motion(renderer: HeadlessRenderer, datum: dict, 
                  filename: str, text_for_vid=None, pose_repr='6d',
                  color=(160 / 255, 160 / 255, 160 / 255, 1.0),
                  return_verts=False, smpl_layer=None) -> None:
    """
    Function to render a video of a motion sequence
    renderer: aitviewer renderer
    datum: dictionary containing sequence of poses, body translations and body orientations
        data could be numpy or pytorch tensors
    filename: the absolute path you want the video to be saved at

    """
    from aitviewer.headless import HeadlessRenderer
    from aitviewer.renderables.smpl import SMPLSequence
    import trimesh
    if isinstance(datum, dict): datum = [datum]
    if not isinstance(color, list): 
        colors = [color] 
    else:
        colors = color
    # assert {'body_transl', 'body_orient', 'body_pose'}.issubset(set(datum[0].keys()))
    # os.environ['DISPLAY'] = ":11"
    gender = 'neutral'
    only_skel = False
    import sys
    seqs_of_human_motions = []
    if smpl_layer is None:
        from aitviewer.models.smpl import SMPLLayer
        smpl_layer = SMPLLayer(model_type='smplh', 
                                ext='npz',
                                gender=gender)
    
    for iid, mesh_seq in enumerate(datum):

        if pose_repr != 'aa':
            global_orient = transform_body_pose(mesh_seq['body_orient'],
                                                f"{pose_repr}->aa")
            body_pose = transform_body_pose(mesh_seq['body_pose'],
                                            f"{pose_repr}->aa")
        else:
            global_orient = mesh_seq['body_orient']
            body_pose = mesh_seq['body_pose']

        body_transl = mesh_seq['body_transl']
        sys.stdout.flush()

        old = os.dup(1)
        os.close(1)
        os.open(os.devnull, os.O_WRONLY)
        
        smpl_template = SMPLSequence(body_pose,
                                     smpl_layer,
                                     poses_root=global_orient,
                                     trans=body_transl,
                                     color=colors[iid],
                                     z_up=True)
        if only_skel:
            smpl_template.remove(smpl_template.mesh_seq)

        seqs_of_human_motions.append(smpl_template)
        renderer.scene.add(smpl_template)
    # camera follows smpl sequence
    # FIX CAMERA
    from sinc.tools.transform3d import get_z_rot
    R_z = get_z_rot(global_orient[0], in_format='aa')
    heading = -R_z[:, 1]
    xy_facing = body_transl[0] + heading*2.5
    camera = renderer.lock_to_node(seqs_of_human_motions[0],
                                    (xy_facing[0], xy_facing[1], 1.5), smooth_sigma=5.0)

    # /FIX CAMERA
    if len(mesh_seq['body_pose']) == 1:
        renderer.save_frame(file_path=str(filename) + '.png')
        sfx = 'png'
    else:
        renderer.save_video(video_dir=str(filename), output_fps=30)
        sfx = 'mp4'

    # aitviewer adds a counter to the filename, we remove it
    # filename.split('_')[-1].replace('.mp4', '')
    # os.rename(filename + '_0.mp4', filename[:-4] + '.mp4')
    if sfx == 'mp4':
        os.rename(str(filename) + f'_0.{sfx}', str(filename) + f'.{sfx}')

    # empty scene for the next rendering
    for mesh in seqs_of_human_motions:
        renderer.scene.remove(mesh)
    renderer.scene.remove(camera)

    sys.stdout.flush()
    os.close(1)
    os.dup(old)
    os.close(old)

    if text_for_vid is not None:
        fname = put_text(text_for_vid, f'{filename}.{sfx}', f'{filename}_.{sfx}')
        os.remove(f'{filename}.{sfx}')
    else:
        fname = f'{filename}.{sfx}'
    return fname
