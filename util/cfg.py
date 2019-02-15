from collections import OrderedDict

config = OrderedDict()
config['label_excel_path'] = 'E:/DataSet/Face/SCUT-FBP/Rating_Collection/AttractivenessLabel.xlsx'
config['face_image_filename'] = 'E:/DataSet/Face/SCUT-FBP/Processed/SCUT-FBP-{0}.jpg'
config['predictor_path'] = "/media/lucasx/Document/ModelZoo/shape_predictor_68_face_landmarks.dat"
config['vgg_face_model_mat_file'] = 'E:/ModelZoo/vgg-face.mat'
config['eccv_dataset_attribute'] = '../preprocess/eccv_face_attribute.csv'
config['eccv_dataset_split_csv_file'] = \
    '/media/lucasx/Document/DataSet/Face/eccv2010_beauty_data_v1.0/eccv2010_beauty_data/eccv2010_split1.csv'
config['scut_fbp5500_img_base_dir'] = "E:/DataSet/Face/SCUT-FBP5500/Crop"
