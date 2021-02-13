"""
Read youcook2 metadata.
"""
import json
from pathlib import Path
import ipdb

from nntrainer import arguments, utils


# map original youcook2 splits to our syntax
SPLIT_MAP = {
    "training": "train",
    "validation": "val"
}

# FIXES = {"tomoatoes": "tomatoes"}
FIXES = {}


def main():
    # argparser
    parser = utils.ArgParser(description=__doc__)
    arguments.add_path_args(parser)
    args = parser.parse_args()

    # setup dataset path
    path_data = Path("data/")
    path_dataset = Path(path_data) / "bksnmovies"
    captions_path = Path("annotations") / "bksnmovies"
    print(f"Working on dataset path {path_dataset} captions from {captions_path}")

    # setup other paths
    meta_file = path_dataset / "meta_all.json"

    # load input meta
    meta_in_file = (captions_path / "bknmovies_v0_split_1_nonempty.json")
    with meta_in_file.open("rt", encoding="utf8") as fh:
        meta_raw = json.load(fh)["database"]

    # load text, video and positives meta
    text_in_file = (captions_path / "text_data.json")
    vid_in_file = (captions_path / "vid_data.json")
    pos_in_file = (captions_path / "pos_data.json")
    
    text_data = json.load(open(text_in_file, 'r'))
    vid_data = json.load(open(vid_in_file, 'r'))
    pos_data = json.load(open(pos_in_file, 'r'))

    # loop all videos in the dataset
    meta_dict = {}
    for key, meta in meta_raw.items():
        # load relevant meta fields
        duration_sec = meta["duration"]
        split = SPLIT_MAP[meta["subset"]]

        # get text and video segments, as well as positive pairs annotations
        text_segs = [{"text":sent} for sent_id, sent in text_data[key].items()]
        vid_segs = [{"start_sec":clip[0], "stop_sec":clip[1]} for clip_id, clip in vid_data[key].items()]

        # get dictionary from sentence id to index in the text_segs list
        sent_id_to_index = {sent_id:index for index, (sent_id, sent) in enumerate(text_data[key].items())}
        clip_id_to_index = {clip_id:index for index, (clip_id, clip) in enumerate(vid_data[key].items())}

        positives = []
        for anno in pos_data[key]:
            bop = []
            for clip_id in anno['positive_shots']:
                for sent_id in anno['positive_sentences']:
                    try:
                        bop.append((sent_id_to_index[str(sent_id)], clip_id_to_index[str(clip_id)]))
                    except:
                        ipdb.set_trace()
            positives.append(bop)

        # create video meta
        meta_dict[key] = {"data_key":key, "duration_sec":duration_sec, "split":split, "text_segments":text_segs, "vid_segments":vid_segs, "positives":positives}

    # write meta to file
    json.dump(meta_dict, meta_file.open("wt", encoding="utf8"), sort_keys=True)
    print(f"wrote {meta_file}")


if __name__ == "__main__":
    main()
