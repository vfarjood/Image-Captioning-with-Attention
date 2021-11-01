# Image-Captioning-with-Attention

This is a command-line program based on python using PyTorch library to annotate pictures. 

All the material will be available here:
https://drive.google.com/drive/folders/1ZSwuc_xGe7RvGgG6gysD7G2KFyTjXUlk?usp=sharing

Architecture: Encoder(CNN-ResNet50), Decoder(LSTM), Attention Network
![image](https://user-images.githubusercontent.com/93528581/139692190-0524c4ee-971f-4299-a01c-2e6f0ed38f9a.png)


------------------------------------------------------------------------------------------------
- Source folder is composed of:

	1. A Python file:\
			- The source code to run the program.
	2. A pdf file:\
			- The report of the project and you can find all the necessary details of the project.

------------------------------------------------------------------------------------------------
- Structure of the code contains:

	- Classes:
		- TextManager,	
		- CapDataset,	
		- Encoder_CNN,	
		- Attention,	
		- Decoder_LSTM,	
		- Encoder_Decoder,	
	- Functions:
		- plot_attention,
		- show_image,
		- save_acc_graph,
		- parse_command_line_arguments,
	- MAIN point:
		- Training,
		- Evaluating,
		- Testing,

------------------------------------------------------------------------------------------------
- Dataset:	

	- Dataset will be available on the link below:\
	https://drive.google.com/drive/folders/1SALkACRkJanBcWuBf_oJY1t_Ys76xTmX?usp=sharing
		
	- It is composed of a folder and a text file:
		1. Folder contains 8,000 images
		2. The Text file compose of 40,000 captions in which for each image we have 5 captions.
	- Each raw of caption.txt contains two elements, image name and image caption, in which they are separated by ‘,’.
	- Dataset is not splitted by default so you will need to split it into train_set, validation_set, test_set.

------------------------------------------------------------------------------------------------
- Instruction to run the code:

	1. In order to run the program first make sure you install all the necessary libraries(you can find them at the beginning of the code).
	2. Make sure you already downloaded the dataset.
	3. It is a command-line base program so you need to open a terminal.
	4. Run the python file with one of the mandatory arguments [’train’,’eval’,’ test’] and the address of dataset folder(This is the minimum requirement).

			- example:	python image_captioning_with_attention.py train flickr8k/
		
	5. If you need help for defining arguments then you can try with ‘-h’ argument at the end of the command.

			- example:	python image_captioning_with_attention.py -h
		
	6. For defining any arbitrary arguments, you need to specify it with the name of the argument. ‘—-argument= ‘

			- example: 	python image_captioning_with_attention.py train —-batch_size=16

	*Note: if you are running the program on ‘Google Colab’ then you need to pass your arguments like this way:
	
			- Change line-883 to:	parsed_arguments = parser.parse_args(['train', '/content/flickr8k'])

------------------------------------------------------------------------------------------------
- Instruction for eval and test:

	1- make sure you have ‘attention_model.pth’ file.\
	2- make sure you have ‘vocabulary.dat’ file.\
	3- you have to provide dataset folder as you did in training procedure:\
			- eval: data must contains ‘Images’ folder and ‘Caption.txt’ file.\
			- test: only provide ‘Images’ folder.

*Note: It is assumed that you only want to evaluate or test the model without training.

------------------------------------------------------------------------------------------------

- If you need any help, please let me know by sending an email to: vahid.farjood@gmail.com

------------------------------------------------------------------------------------------------



