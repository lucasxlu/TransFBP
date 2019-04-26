from collections import OrderedDict

config = OrderedDict()
config['scut_fbp_base'] = '/home/xulu/DataSet/Face/SCUT-FBP'
config['test_ratio'] = 0.2
config['label_excel_path'] = '/home/xulu/DataSet/Face/SCUT-FBP/Rating_Collection/AttractivenessLabel.xlsx'
config['face_image_filename'] = '/home/xulu/DataSet/Face/SCUT-FBP/Crop/SCUT-FBP-{0}.jpg'
config['predictor_path'] = "/media/lucasx/Document/ModelZoo/shape_predictor_68_face_landmarks.dat"
config['vgg_face_model_mat_file'] = '/home/xulu/ModelZoo/vgg-face.mat'
config['eccv_dataset_attribute'] = '../preprocess/eccv_face_attribute.csv'
config[
    'eccv_dataset_split_csv_file'] = '/home/xulu/DataSet/Face/eccv2010_beauty_data_v1.0/eccv2010_beauty_data/eccv2010_split1.csv'
config['scut_fbp5500_img_base_dir'] = "/home/xulu/DataSet/Face/SCUT-FBP5500/Crop"
