# Code for paper "Towards Fair and Robust Classification"

## General notes (**important**):
- Environments and additional packages:
	- Python 3.7.9
	- PyTorch
	- AIF360
- Due to further change to the paper, for all codes below, the corresponding experiment is referred to "paper.pdf" in current directory as a snapshot;
- About datasets:
	- Datasets are stored in ``./data/`` folder;
	- A dataset (e.g., ``adult`` for other datasets replace adult with its name string) should be stored in 3 ``.csv`` files:
		- ``adult.csv``: The whole dataset
		- ``adult_train.csv`` The dataset with pre-shuffled and pre-splitted training dataset;
		- ``adult_test.csv`` The dataset with pre-shuffled and pre-splitted testing dataset;
		- ``adult.pkl.gz`` The gzipped pickle binary file of the dataset with AIF360 format;
		- ``adult_train.pkl.gz`` and ``adult_test.pkl.gz`` are the gzipped pickle binary file of ``adult_train.csv`` and ``adult_test.csv``
		- ``*.pkl.gz`` files are only used for running existing approaches in AIF360, ``.csv`` files are used for other experiments;
		- For new dataset: Generate .csv file, split to training and testing dataset, and add to ``./data/`` folder, for all .csv file, label must be the last column; Then generate corresponding ``.pkl.gz`` files using ``.csv`` file.
- All experimental scripts are divide into 3 code blocks:
	- Implementation: The code that implements the experiment with specific algorithm, these script takes arguments of a single setting from command line and only execute and output the result for that setting. For example ``python InProcess.py adult race FGSM 0.1 0.4`` will execute in-processing algorithm with adult as the dataset, race as the senstive attribute, FGSM as the robustness against, lambda_R=0.1 and lambda_F=0.4.
	- Batching script: These scripts has prefix ``exec_``, e.g., ``exec_InProcess.py``, which are used to enmuerate all settings for the implementation to execute. The scripts can resume the progress, i.e., it can automatically detect which settings have been exeucted, skip them and start from the next uncalulated setting.
	- Result script: These scripts convert the raw output of the implementation into parsable results, e.g., tabular data, pdf figures, etc.
	- ** In a word, the implementation part can be ignored unless there are some erros, and the execution sequence is: (1) Run batching scripts; and (2) Run result scripts and get the result. **


## Existing Approaches (Figure 2 & Figure 8)

- Implementation: ``ExistingCombosFnR.py``
- Batching script: ``exec_ExistingCombosFnR.py``
- Result script: ``./result/existings/parse_FnR.py``, the output contains three lines and need to be filled in the pgfplot script in corrseponding ``.tex`` file in the paper source.

## Heatmap of in-processing (Figure 3)

- Implementation: ``InProcess.py``
- Batching script: ``exec_InProcess.py``
- Result script: ``./result/inproc/parse_Heatmap.py``

## Angle of in-processing gradient (Figure 4 & Figure 9)

- Implementation: ``InProcess.py``
- Batching script: ``exec_InProcess.py``
- Result script: ``./result/inproc/parse_Angle.py``, the output contains one line and need to be filled in the pgfplot script in corrseponding ``.tex`` file in the paper source.

## Heatmap of pre-processing (Figure 5 & Figure 10)

- Implementation: ``PreProcessOld.py`` for the old version, and ``PreProcessInflu.py`` for the new version with influence function on robustness as well
- Batching script: ``exec_PreProcess.py``
- Result script: ``./result/preproc/parse_Heatmap.py``

## Downstream models (Figure 6 & Figure 11)

- Implementation: ``PreProcessOld.py`` for the old version, and ``PreProcessInflu.py`` for the new version with influence function on robustness as well. Need to turn the argv ``saveflag`` to ``True``. This is to generate pre-processed data;
- Batching script: ``exec_PreProcGen.py`` for generating all data, and ``PreProcessDownstreams.py`` for running pre-set models, get the result and store the result for different downstream models in files
- Result script: ``./result/preproc/parse_Figures.py``

## Compare pre- and in-processing (Figure 7 & Figure 12)

- Implementation: N/A, you need to get the result of pre- and in-processing first by running the heatmap parsing script of pre- and in-processing ``./result/preproc/parse_Heatmap.py`` and ``./result/inproc/parse_Heatmap.py``. After running these heatmap scripts, the data in those heatmap will be autommatically saved to ``./result/preindiff/`` folder as a ``.txt`` file;
- Batching script: N/A
- Result script: The result script of pre- and in-processing will automatically generate the heatmap data in ``./result/preindiff/``, so after the generation, run ``./result/preindiff/draw_diff.py``

## Original fairness & robustness (Table 2)

- Implementation: ``ExistingCombosFnR.py``
- Batching script: ``exec_ExistingCombosFnR.py``

You can get the result from the result of existing approaches, just set wF=0 and wR=0, and the result will be original.

## Fairness on robustenss (Table 3)

- Implementation: ``ExistingCombosFnR.py``
- Batching script: ``exec_ExistingCombosFnR.py``
- Result script: ``./result/existings/parse_F2R.py``

You can get the result from the result of existing approaches, just compare the robustness score (but keep wR=0) of the setting with specific fairness wF parameter with original setting (wF=0 and wR=0).

## Robustness on fairness (Table 4)

- Implementation: ``ExistingCombosFnR.py``
- Batching script: ``exec_ExistingCombosFnR.py``
- Result script: ``./result/existings/parse_R2F.py``

You can get the result from the result of existing approaches, just compare the fairness score (but keep wF=0) of the setting with specific robustness wR parameter with original setting (wF=0 and wR=0).

