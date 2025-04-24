# exposed functions to run asoid via lines
import os
import numpy as np
from configparser import ConfigParser
import pandas as pd

import streamlit as st
import logging

streamlit_loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict if name.startswith("streamlit")]
for logger in streamlit_loggers:
    logger.setLevel(logging.ERROR)

from asoid.utils.auto_active_learning import get_available_conf_options, RF_Classify
from asoid.utils.extract_features import Extract

from asoid.utils.predict import bsoid_predict_numba_noscale, bsoid_predict_proba_numba_noscale, weighted_smoothing
from asoid.utils.extract_features_2D import feature_extraction
from asoid.utils.extract_features_3D import feature_extraction_3d
from asoid.utils.load_workspace import load_new_pose, load_iterX
from asoid.utils.import_data import load_pose
from asoid.utils.preprocessing import adp_filt, sort_nicely

from asoid.utils.reporting import extract_descriptors, prep_labels_single

from tqdm import tqdm



def load_config(config_path, verbose = True):
    """
    Load the configuration file.
    :param config_path: Path to the configuration file.
    :return: Configuration object.
    """
    config = ConfigParser()
    config.read(config_path)

    if verbose:
        # report config
        for key, value in config.items():
            print(f"{key}: ")
            for k, v in value.items():
                print(f"  {k}: {v}")


    return config


def extract_features(config, duration_min: float = 0.1, verbose = True):
    """ Extract features from the data.
    :param config: Configuration object.
    :param duration_min: Duration of the features in seconds.
    """
    assert duration_min > 0, "Duration must be greater than 0"
    working_dir = config["Project"].get("PROJECT_PATH")
    prefix = config["Project"].get("PROJECT_NAME")
    is_3d = config["Project"].getboolean("IS_3D")
    annotation_classes = [x.strip() for x in config["Project"].get("CLASSES").split(",")]
    framerate = config["Project"].getfloat("FRAMERATE")
    
    frames2integ = round(framerate * (duration_min / 0.1))
    if verbose:
        print(f"Frames to integrate to reach {duration_min} sec: {frames2integ}")
    extractor = Extract(working_dir, prefix, frames2integ, is_3d)
    extractor.main()


def train_model(config
    , init_ratio: float
    , max_iter: int
    , max_samples_iter: int
    , iteration = None
    , conf_type = None
    , conf_threshold = None
    , mode = "notebook"):
    """ Train the model.
    :param config: Configuration object.
    :param init_ratio: Initial ratio of training data.
    :param max_iter: Maximum number of iterations.
    :param max_samples_iter: Maximum number of samples per iteration.
    :param iteration: Iteration number. Default is None, which means the last iteration.
    :param conf_type: Confidence type. Default is None, which means the last confidence type.
    :param conf_threshold: Confidence threshold.
    :param mode: Mode of operation (notebook or cli).
    :return: None
    """


    assert init_ratio > 0 and init_ratio < 1, "Initial ratio must be between 0 and 1"
    assert max_iter > 0, "Max iterations must be greater than 0"
    assert max_samples_iter > 0, "Max samples per iteration must be greater than 0"
    assert conf_type in get_available_conf_options().keys(), f"Conf type must be one of {get_available_conf_options().keys()}"
    assert conf_threshold > 0 and conf_threshold < 1, "Conf threshold must be between 0 and 1"
    assert mode in ["notebook", "cli"], "Mode must be either notebook or cli"

    working_dir = config["Project"].get("PROJECT_PATH")
    prefix = config["Project"].get("PROJECT_NAME")
    annotation_classes = [x.strip() for x in config["Project"].get("CLASSES").split(",")]
    software = config["Project"].get("PROJECT_TYPE")
    exclude_other = config["Project"].getboolean("EXCLUDE_OTHER")
    train_fx = config["Processing"].getfloat("TRAIN_FRACTION")
    #TODO: needs failsave for older projects
    if conf_type is None:
        try:
            conf_type = config["Processing"].get("CONF_TYPE")
        except KeyError:
            conf_type = get_available_conf_options().keys()[0]
    if conf_threshold is None:
        try:
            conf_threshold = config["Processing"].getfloat("CONF_THRESHOLD")
        except KeyError:
            conf_threshold = 0.5
    
    if iteration is None:
        try:
            iteration = config["Processing"].getint("ITERATION")
        except KeyError:
            iteration = 0
    project_dir = os.path.join(working_dir, prefix)
    iter_folder = str.join('', ('iteration-', str(iteration)))

    os.makedirs(os.path.join(project_dir, iter_folder), exist_ok=True)
    # 
    rf_classifier = RF_Classify(working_dir, prefix, iter_folder, software
                            , init_ratio
                            , max_iter
                            , max_samples_iter
                            , annotation_classes
                            , exclude_other
                            , conf_type
                            , conf_threshold
                            , mode = mode)
    rf_classifier.main()

def _convert_predictions(predictions, annotation_classes, framerate):
    """takes numerical labels and transforms back into one-hot encoded file (BORIS style).
    :param predictions: list of predictions
    :param annotation_classes: list of annotation classes
    :param framerate: framerate of the video
    :return: DataFrame of predictions"""
    # convert to pandas dataframe
    df = pd.DataFrame(predictions, columns=["labels"])
    time_clm = np.round(np.arange(0, df.shape[0]) / framerate, 2)
    # convert numbers into behavior names
    class_dict = {i: x for i, x in enumerate(annotation_classes)}
    df["classes"] = df["labels"].copy()
    for cl_idx, cl_name in class_dict.items():
        df["classes"].iloc[df["labels"] == cl_idx] = cl_name

    # for simplicity let's convert this back into BORIS type file
    dummy_df = pd.get_dummies(df["classes"]).astype(int)
    # add 0 columns for each class that wasn't predicted in the file
    not_predicted_classes = [x for x in annotation_classes if x not in np.unique(df["classes"].values)]
    for not_predicted_class in not_predicted_classes:
        dummy_df[not_predicted_class] = 0

    dummy_df["time"] = time_clm
    dummy_df = dummy_df.set_index("time")
    
    return dummy_df

def _save_predictions(predictions_df, output_filename):
    """ Save the predictions to a file.
    :param predictions_df: DataFrame of predictions.
    :param output_filename: Name of the output file.
    :return: None
    """
    # save predictions
    predictions_df.to_csv(output_filename, index=False)
    print(f"Predictions saved to {output_filename}")

def _report(pred_df, annotation_classes, framerate):
    """
    This function generates a report from the predictions.
    :param pred_df: DataFrame of predictions.
    :return: None
    """
    
    count_df = extract_descriptors(pred_df, annotation_classes, framerate)
    print("Behavior report:")
    print(count_df)

# prediction on new data

def predict(path_list, config, smooth_size = 0, save_predictions = True, verbose = True):
    """ Predict the behavior from pose data and returns raw and smoothed output.
    :param path_list: List of paths to pose files.
    :param config: Configuration object.
    :param smooth_size: Size of the smoothing window. If 0, no smoothing is applied. Default is 0.
    :param save_predictions: If True, save the predictions to csv file in BORIS style at location. Default is True.
    :return: predictions_raw, predictions_match: List of predictions per file (raw and smoothed).
    """
    assert smooth_size > 0, "Smooth size must be greater than 0"
  
    # get parameters from config
    working_dir = config["Project"].get("PROJECT_PATH")
    prefix = config["Project"].get("PROJECT_NAME")
    annotation_classes = [x.strip() for x in config["Project"].get("CLASSES").split(",")]
    software = config["Project"].get("PROJECT_TYPE")
    ftype = [x.strip() for x in config["Project"].get("FILE_TYPE").split(",")]
    selected_bodyparts = [x.strip() for x in config["Project"].get("KEYPOINTS_CHOSEN").split(",")]
    is_3d = config["Project"].getboolean("IS_3D")
    multi_animal = config["Project"].getboolean("MULTI_ANIMAL")
    llh_value = config["Processing"].getfloat("LLH_VALUE")
    iteration = config["Processing"].getint("ITERATION")
    framerate = config["Project"].getint("FRAMERATE")
    duration_min = config["Processing"].getfloat("MIN_DURATION")
    project_dir = os.path.join(working_dir, prefix)
    iter_folder = str.join('', ('iteration-', str(iteration)))
    os.makedirs(os.path.join(project_dir, iter_folder), exist_ok=True)
    frames2integ = round(float(framerate) * (duration_min / 0.1))

    #load model
    [iterX_model, _, _] = load_iterX(project_dir, iter_folder) 

    # run prediction

    features = [None]
    predict_arr = None
    predictions_match = None

    new_pose_csvs = path_list
    repeat_n = int(frames2integ / 10)
    total_n_frames = []

    # extract features, bin them
    features = []
    # for i, data in enumerate(processed_input_data):
    for i, f in enumerate(tqdm(new_pose_csvs, desc="Extracting spatiotemporal features from pose")):
    # for i, f in enumerate(new_pose_csvs):

        current_pose = load_pose(f, software, multi_animal)
        bp_level = 1
        bp_index_list = []

        if i == 0:
            # check if all bodyparts are in the pose file

            if len(selected_bodyparts) > len(current_pose.columns.get_level_values(bp_level).unique()):
                raise ValueError(f'Not all selected keypoints/bodyparts are in the pose file: {f.name}')

            elif len(selected_bodyparts) < len(current_pose.columns.get_level_values(bp_level).unique()):
                # subselection would take care of this, so we need to make sure that they all exist
                for bp in selected_bodyparts:
                    if bp not in current_pose.columns.get_level_values(bp_level).unique():
                        raise ValueError(f'At least one keypoint "{bp}" is missing in pose file: {f.name}')
        
            for bp in selected_bodyparts:
                bp_index = np.argwhere(current_pose.columns.get_level_values(bp_level) == bp)
                bp_index_list.append(bp_index)
            selected_pose_idx = np.sort(np.array(bp_index_list).flatten())

            # get likelihood column idx directly from dataframe columns
            idx_llh = [i for i, s in enumerate(current_pose.columns) if "likelihood" in s and s in current_pose.columns[selected_pose_idx]]

            # the loaded sleap file has them too, so exclude for both
            idx_selected = [i for i in selected_pose_idx if i not in idx_llh]

        # filtering does not work for 3D yet
        # check if there is a z coordinate

        if "z" in current_pose.columns.get_level_values(2):
            if is_3d is not True:
                raise ValueError("3D data detected. But parameter is set to 2D project.")
            print("3D project detected. Skipping likelihood adaptive filtering.")
            # if yes, just drop likelihood columns and pick the selected bodyparts
            filt_pose = current_pose.iloc[:, idx_selected].values
        else:
            filt_pose, _ = adp_filt(current_pose, idx_selected, idx_llh, llh_value)

        # using feature scaling from training set
        if not is_3d:
            feats, _ = feature_extraction([filt_pose], 1, frames2integ)
        else:
            feats, _ = feature_extraction_3d([filt_pose], 1, frames2integ)

        total_n_frames.append(filt_pose.shape[0])
        features.append(feats)

    coll_predictions_raw = []
    coll_predictions_match = []
    for i in tqdm(range(len(features)), desc="Behavior prediction from spatiotemporal features"):

        predict = bsoid_predict_numba_noscale([features[i]], iterX_model)
        # pred_proba = bsoid_predict_proba_numba_noscale([features[i]], iterX_model)
        predict_arr = np.array(predict).flatten()

        predictions_raw = np.pad(predict_arr.repeat(repeat_n), (repeat_n, 0), 'edge')[:total_n_frames[i]]
        if smooth_size > 0:
            # smooth predictions
            predictions_match = weighted_smoothing(predictions_raw, size=smooth_size)
        else:
            predictions_match = predictions_raw

        coll_predictions_raw.append(predictions_raw)
        coll_predictions_match.append(predictions_match)

        pred_df = _convert_predictions(predictions_match, annotation_classes, framerate)

        if verbose:
            # report predictions
            
            prep_df = prep_labels_single(pred_df, annotation_classes)
            _report(prep_df, annotation_classes, framerate)

        if save_predictions:
            # save predictions
            curr_file_name = new_pose_csvs[i]
            curr_file_name = os.path.basename(curr_file_name)
            curr_file_name = os.path.splitext(curr_file_name)[0]
            output_filename = os.path.join(project_dir, iter_folder, curr_file_name + "_predictions.csv")

            _save_predictions(pred_df, output_filename)

      

    return coll_predictions_raw, coll_predictions_match

  


    