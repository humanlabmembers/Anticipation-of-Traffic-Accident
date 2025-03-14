First of all, thank you for your attention to our work. To reproduce the results of this experiment, please follow the steps below:

1. Install the environment

Please install the Conda environment according to the dependencies in 'requirements'.

2. Download the dataset

Due to the anonymity rule, we cannot directly provide an external link to the dataset. Please download all the videos of the DAD dataset according to the guidance of the reference. Please divide the training set and test set according to the settings of the dataset.

3. Feature extraction

Please use the Python code in 'Util_scripts' to perform object detection, depth and video feature extraction on the video respectively.

4. Path modification

Please modify the path of the 'main' file and 'Dataloader' file according to your personal settings.

5. Checkpoints

We plan to provide 4 checkpoints in the ckpts folder:

	Ⅰ. DAD.pth
	Experimental results using the original DAD dataset.

	Ⅱ. DAD+20%.pth
	Experimental results using the DAD dataset and adding 20% ​​to the training set to generate videos.

	Ⅲ. DAD+40%
	Experimental results using the DAD dataset and adding 40% to the training set to generate videos.

	IV. DAD_replaced40%.pth
	The experimental results of using the DAD dataset and randomly replacing the original videos with 40% of the generated videos in the training set.

However, due to file size limitation, we can unfortunately only provide the code with ckpt. If you are interested in our work, please contact us at a later stage to obtain the corresponding ckpt.

6. Run the test
Change '--model_file' to the checkpoint path you want to test, set '--phase' to 'test' and run the 'main' file to get the output results.