import argparse
import logging
import multiprocessing as mp
import os
import shutil
import subprocess as sp
from functools import partial

import nibabel as nib
import numpy as np
import pandas as pd
from radiomics import featureextractor



MODALITIES = ["T1", "CT1", "T2", "FLAIR"]
PREPROCESSED_SUFFIX = "_r2s_bet_reg.nii.gz"

LABELS_HDGLIOAUTO = {
    1: "Non-enhancing",
    2: "Contrast-enhancing",
}

RADIOMICS_FILES = [
    "CT1_r2s_bet_reg.nii.gz",
    "T1_r2s_bet_reg.nii.gz",
    "T2_r2s_bet_reg.nii.gz",
    "FLAIR_r2s_bet_reg.nii.gz",
]


def run_cmd(cmd, verbose=False):
    """
    Runs a command in a subprocess and returns its output.

    Parameters
    ----------
    cmd : list
        List of strings representing the command to be run.
    verbose : bool, optional
        If True, prints the command before running it.

    Returns
    -------
    output : bytes
        The output of the command.
    """
    if verbose:
        print(" ".join(cmd))
    return sp.check_output(cmd)


def reorient_to_std(file_, overwrite=False):
    """
    Reorients a NIfTI image to the standard orientation.

    Parameters
    ----------
    file_ : str
        Path to the input NIfTI image.
    overwrite : bool, optional
        If True, overwrite the output file if it already exists.

    Returns
    -------
    out : str
        Path to the reoriented output NIfTI image.
    """
    out = file_.replace(".nii.gz", "_r2s.nii.gz")
    if overwrite or not os.path.exists(out):
        run_cmd(["fslreorient2std", file_, out])
    return out


def register_to_ref(file_, ref, overwrite=False):
    """
    Registers an image to a reference using FLIRT.

    Parameters
    ----------
    file_ : str
        Path to the input NIfTI image.
    ref : str
        Path to the reference NIfTI image.
    overwrite : bool, optional
        If True, overwrite the output file if it already exists.

    Returns
    -------
    out : str
        Path to the registered output NIfTI image.
    """
    out = file_.replace(".nii.gz", "_reg.nii.gz")
    mat = file_.replace(".nii.gz", "_reg.mat")
    if overwrite or not os.path.exists(out):
        run_cmd([
            "flirt", "-in", file_, "-ref", ref,
            "-out", out, "-dof", "6",
            "-interp", "spline", "-omat", mat
        ])
    return out

def preprocess_modalities(inputs, output_dir, overwrite=False, verbose=False):
    """
    Preprocesses a list of modalities by reorienting them to the standard orientation, 
    registering them to the modality with the smallest spacing, and saving the 
    resulting registered images and their corresponding masks.

    Parameters
    ----------
    inputs : list of str
        List of paths to the input NIfTI images.
    output_dir : str
        Path to the output directory.
    overwrite : bool, optional
        If True, overwrite the output files if they already exist.
    verbose : bool, optional
        If True, prints the commands before running them.

    Returns
    -------
    reg_files : list of str
        List of paths to the registered output NIfTI images.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)

    processed = []
    for name, path in zip(MODALITIES, inputs):
        target = os.path.join(output_dir, f"{name}.nii.gz")
        if overwrite or not os.path.exists(target):
            shutil.copy(path, target)
        processed.append(target)

    with mp.Pool(min(len(processed), mp.cpu_count())) as pool:
        processed = pool.map(partial(reorient_to_std, overwrite=overwrite), processed)

    spacings = [np.prod(nib.load(f).header.get_zooms()) for f in processed]
    ref_idx = int(np.argmin(spacings))

    bet_files, masks = [], []
    for f in processed:
        bet = f.replace(".nii.gz", "_bet.nii.gz")
        mask = f.replace(".nii.gz", "_bet_bet.nii.gz")
        if overwrite or not os.path.exists(bet):
            run_cmd(["hd-bet", "-i", f, "-o", bet, "--save_bet_mask"], verbose)
            run_cmd(["fslmaths", bet, "-mas", mask, bet], verbose)
        bet_files.append(bet)
        masks.append(mask)

    shutil.copy(masks[ref_idx], "mask.nii.gz")

    with mp.Pool(min(len(bet_files), mp.cpu_count())) as pool:
        reg_files = pool.map(
            partial(register_to_ref, ref=bet_files[ref_idx], overwrite=overwrite),
            bet_files,
        )

    return reg_files


def run_segmentation(files, overwrite=False, verbose=False):
    """
    Runs the segmentation algorithm using the input files and saves the result to "segmentation.nii.gz".

    Parameters
    ----------
    files : list of str
        List of paths to the input NIfTI images.
    overwrite : bool, optional
        If True, overwrite the output file if it already exists.
    verbose : bool, optional
        If True, prints the command before running it.
    """
    if overwrite or not os.path.exists("segmentation.nii.gz"):
        run_cmd([
            "hd_glio_predict",
            "-t1", files[0],
            "-t1c", files[1],
            "-t2", files[2],
            "-flair", files[3],
            "-o", "segmentation.nii.gz",
        ], verbose)

        img = nib.load("segmentation.nii.gz")
        data = img.get_fdata()
        data[data == 3] = 0
        nib.save(nib.Nifti1Image(data, img.affine), "segmentation.nii.gz")

def extract_radiomics(output_dir, param_file):
    """
    Extracts radiomic features from the input images and saves the results to a CSV file.

    Parameters
    ----------
    output_dir : str
        Path to the output directory.
    param_file : str
        Path to the parameter file for the feature extractor.

    Returns
    -------
    None
    """
    extractor = featureextractor.RadiomicsFeatureExtractor(param_file)
    results = []

    for img in RADIOMICS_FILES:
        for label, name in LABELS_HDGLIOAUTO.items():
            info = {
                "Image": img,
                "Mask": "segmentation.nii.gz",
                "Label": label,
                "Label name": name,
                "Sequence": img.split("_")[0],
            }
            try:
                feats = extractor.execute(img, "segmentation.nii.gz", label=label)
                results.append({**info, **feats})
            except Exception:
                results.append(info)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "hdglioauto_radiomics_features.csv"), index=False)


def main():
    """
    Main function of the program.

    It parses the command line arguments using argparse, sets the environment variable
    CUDA_VISIBLE_DEVICES to the specified device, and calls the preprocess_modalities,
    run_segmentation, and extract_radiomics functions.

    Parameters
    ----------
    -i, --input_dir : str
        Path to the input directory containing the 4 input modalities.
    -o, --output_dir : str, optional
        Path to the output directory where the results will be saved.
    --overwrite : bool, optional
        If True, overwrite the output files if they already exist.
    --verbose : bool, optional
        If True, print the commands before running them.
    --device : str, optional
        Index of the CUDA device to use.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", required=True)
    parser.add_argument("-o", "--output_dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    output_dir = args.output_dir or args.input_dir

    inputs = [
        os.path.join(args.input_dir, f"{m}.nii.gz")
        for m in MODALITIES
    ]

    files = preprocess_modalities(inputs, output_dir, args.overwrite, args.verbose)
    run_segmentation(files, args.overwrite, args.verbose)
    extract_radiomics(output_dir, os.path.join(os.getcwd(), "Params.yaml"))


if __name__ == "__main__":
    main()
