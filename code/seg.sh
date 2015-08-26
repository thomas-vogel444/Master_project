python run_experiment.py

varying_number_of_hidden_units/1000








python run_segmentation.py -m ../experimental_results/varying_number_of_convolutional_layers/1_conv_layer -s ../datasets/segmentation_dataset_for_14040204
python run_segmentation.py -m ../experimental_results/varying_number_of_convolutional_layers/2_conv_layers -s ../datasets/segmentation_dataset_for_14040204
python run_segmentation.py -m ../experimental_results/varying_number_of_convolutional_layers/3_conv_layers -s ../datasets/segmentation_dataset_for_14040204
python run_segmentation.py -m ../experimental_results/varying_number_of_convolutional_layers/4_conv_layers -s ../datasets/segmentation_dataset_for_14040204

python run_segmentation.py -m ../experimental_results/varying_number_of_connected_layers/1_connected_layer -s ../datasets/segmentation_dataset_for_14040204
python run_segmentation.py -m ../experimental_results/varying_number_of_connected_layers/2_connected_layers -s ../datasets/segmentation_dataset_for_14040204
python run_segmentation.py -m ../experimental_results/varying_number_of_connected_layers/3_connected_layers -s ../datasets/segmentation_dataset_for_14040204

python run_segmentation.py -m ../experimental_results/varying_number_of_feature_maps/starting_with_16 -s ../datasets/segmentation_dataset_for_14040204
python run_segmentation.py -m ../experimental_results/varying_number_of_feature_maps/starting_with_32 -s ../datasets/segmentation_dataset_for_14040204
python run_segmentation.py -m ../experimental_results/varying_number_of_feature_maps/starting_with_64 -s ../datasets/segmentation_dataset_for_14040204
python run_segmentation.py -m ../experimental_results/varying_number_of_feature_maps/starting_with_128 -s ../datasets/segmentation_dataset_for_14040204

python run_segmentation.py -m ../experimental_results/varying_number_of_hidden_units/100_hidden_units -s ../datasets/segmentation_dataset_for_14040204
python run_segmentation.py -m ../experimental_results/varying_number_of_hidden_units/200_hidden_units -s ../datasets/segmentation_dataset_for_14040204
python run_segmentation.py -m ../experimental_results/varying_number_of_hidden_units/500_hidden_units -s ../datasets/segmentation_dataset_for_14040204
python run_segmentation.py -m ../experimental_results/varying_number_of_hidden_units/1000_hidden_units -s ../datasets/segmentation_dataset_for_14040204
