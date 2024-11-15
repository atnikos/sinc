import os
import json
import random
import streamlit as st

DATASET_FILENAME = 'sinc_synth_2.json'
SPLITS_FILENAME = 'sinc_synth_2_splits.json'


def calculate_lehvenstein_distance(s1, s2):
    if len(s1) < len(s2):
        return calculate_lehvenstein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


# read dataset json file
dataset = json.load(open(os.path.join(os.path.dirname(__file__), DATASET_FILENAME)))
# read splits json file
splits = json.load(open(os.path.join(os.path.dirname(__file__), SPLITS_FILENAME)))


st.set_page_config(layout="wide", page_title="Motion Editing Sinc Synthetic data exploration")
st.markdown("## Motion Editing Sinc Synthetic data exploration")
st.markdown("#### ")
st.write(f"#### Loaded {len(dataset)} annotations from '{DATASET_FILENAME}'.")


# Check if the necessary keys are in session state, initialize if not
if 'dataID' not in st.session_state:
    st.session_state['dataID'] = ""
if 'randomID' not in st.session_state:
    st.session_state['randomID'] = ""

columns = st.columns(2)
# Text input for the annotation ID
with columns[0]:
    dataID_input = st.text_input("Annotation ID:", value=st.session_state.dataID, max_chars=19)
    # Button to search based on the text input
    if st.button('Search ID'):
        # Update the dataID when search is clicked
        if 'sinc_synth_' in dataID_input:
            st.session_state.dataID = dataID_input
        else:
            st.session_state.dataID = 'sinc_synth_' + dataID_input

    search_input = st.text_input("Text search:", value="", max_chars=100)
    if st.button('Search text'):
        # if search_input is 1 non-empty word
        search_input = search_input.strip()
        if search_input and " " not in search_input:
            dataset_items = list(dataset.items())
            random.shuffle(dataset_items)
            # Iterate over the shuffled list to find a match
            for annotation_id, annotation in dataset_items:
                if search_input.lower() in annotation["annotation"].lower().split(' '):
                    st.session_state.dataID = annotation_id
                    break
        elif search_input:
            # get best levenshtein distance match
            best_match = None
            best_distance = float('inf')
            for annotation_id, annotation in dataset.items():
                distance = calculate_lehvenstein_distance(search_input.lower(), annotation["annotation"].lower())
                if distance < best_distance:
                    best_match = annotation_id
                    best_distance = distance
            st.session_state.dataID = best_match

# Show buttons to randomize from splits
with columns[1]:
    for split in splits:
        if st.button(f"Random from split '{split}'"):
            # Randomize an elem from list splits[split] and update session state
            st.session_state.randomID = random.choice(splits[split])
            # Clear the text input field by resetting dataID in the session state
            st.session_state.dataID = ""

# Determine which ID to display: prioritize manual input if available
displayID = st.session_state.dataID if st.session_state.dataID else st.session_state.randomID

# Display the selected or searched ID
if displayID in dataset:
    st.session_state.dataID = ""

    st.write(f"")
    # st.write(f'**{dataset[displayID]["annotation"]}**')
    # above, but bigger and middle-aligned
    st.markdown(f"<div style='text-align: center; font-size: 2em;'>{dataset[displayID]['annotation']}</div>", unsafe_allow_html=True)
    st.write(f"")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.video(str(dataset[displayID]['motion_a']), format='video/mp4', start_time=0)
        # render video as html instead, with autoplay and loop
        # st.write(f'<video style="width: 25vw" controls autoplay loop><source src="{dataset[displayID]["motion_a"]}" type="video/mp4"></video>', unsafe_allow_html=True)
    with col2:
        st.video(str(dataset[displayID]['motion_b']), format='video/mp4', start_time=0)
        #st.write(f'<video style="width: 25vw" controls autoplay loop><source src="{dataset[displayID]["motion_b"]}" type="video/mp4"></video>', unsafe_allow_html=True)
    with col3:
        st.video(str(dataset[displayID]['motion_overlaid']), format='video/mp4', start_time=0)
        #st.write(f'<video style="width: 25vw" controls autoplay loop><source src="{dataset[displayID]["motion_overlaid"]}" type="video/mp4"></video>', unsafe_allow_html=True)
    st.write(f"")

    st.markdown(f"<div style='text-align: center;'>Annotation ID: <b>{displayID}</b></div>", unsafe_allow_html=True)
    # st.write(f'Annotation ID: **{displayID}**')

    st.write(f"")
    st.write(f"---")
    # st.write(f"")
    st.write(dataset[displayID])
else:
    st.write("This annotation ID does not exist or has not been selected.")