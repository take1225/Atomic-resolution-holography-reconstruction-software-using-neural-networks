"Copyright (C) S.Takeuchi 2021 All Rights Reserved."

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import api_3d_air_image
import random
import concurrent.futures
import os

#インスタンス作製
api=api_3d_air_image.Api()
api.connect()

"Atomic Image Reconstruction-Neural Network(AIR-NN)"
#ニューラルネットワーク用データセットの作成クラス(散乱パターン関数で学習データの作成)
class Dataset_from_AIR_NN(Dataset):
    def __init__(self,r_min=float(1.5),r_max=float(10)):
        self._transform=transforms.Compose([transforms.ToTensor()])
        self._atomic_distance_min=r_min
        self._atomic_distance_max=r_max
        self.number_of_atoms=1
        self.scattering_data_range_pix=180
        self.chose_multi_process=True
        self.only_learning_existence_atom=True
        self.angular_direction=False
        self.angle_max=90
        self._number_cpu=os.cpu_count()

    def __call__(self):
        from matplotlib import pyplot as plt
        self.get_scattering_pattern_function()
        self.pre_calculate_parameters_for_various_calculations()
        image=self._convert_from_scattering_pattern_to_orthographic(10)
        self._circular_integral_orthographic(image)
        out_data,out_label=self._make_train_and_test_data_singleprocess(10)
        plt.plot(range(len(out_data[0][0])),out_data[0][0])
        plt.show()
        plt.plot(self._atomic_distance,out_label)
        plt.show()

    #len()としたときの操作
    def __len__(self):
        return self.number_of_atoms

    #data[]みたいにしたときの操作
    def __getitem__(self,id):
        if self.chose_multi_process:
            out_data,out_label=self._make_train_and_test_data_multiprocess(id)
        else:
            out_data,out_label=self._make_train_and_test_data_singleprocess(id)
        return out_data,out_label

    #散乱パターン関数を取得
    def get_scattering_pattern_function(self):
        #画像の選択
        print("Please chose scattering pattern function")
        Enter=str(input("適当な数値を入力="))
        scatt=api.window_get_focused_wid()
        scatterring_image=api.image_get(scatt)
        scatterring_image_caption=api.window_caption_get(scatt)
        print(f'caption of scatterring image:{scatterring_image_caption}')
        scatterring_image_transpose=np.array(scatterring_image.T)
        print(len(scatterring_image_transpose[0]))
        self._sp_image=np.delete(scatterring_image_transpose,np.s_[:self.scattering_data_range_pix],axis=1)
        self._atomic_distance=self._range_revise(scatterring_image[0],self._atomic_distance_min,self._atomic_distance_max)
        self.output_layer=len(self._atomic_distance)

    #任意のデータ範囲に修正
    def _range_revise(self,data,min_range,max_range):
        width=max_range-min_range
        step=width/(len(data)-1)
        range_revise=np.arange(min_range,max_range+step,step)
        return range_revise

    #各種計算のパラメーターをあらかじめ計算
    def pre_calculate_parameters_for_various_calculations(self):
        self._orthographic=np.zeros([len(self._sp_image[0]),len(self._sp_image[0])])
        Cx=int((len(self._sp_image[0])-1)/2)
        Cx_Cy=np.array([int((len(self._sp_image[0])-1)/2),int((len(self._sp_image[0])-1)/2)])
        x=np.arange(0,len(self._orthographic),1)
        y=np.arange(0,len(self._orthographic[0]),1)
        X,Y=np.meshgrid(x,y)
        distance_from_center=np.sqrt((X-Cx_Cy[0])**2+(Y-Cx_Cy[1])**2)
        self._binarization_image=np.where(Cx>=distance_from_center,1,0)
        number_of_zero_pixels=np.sum(self._binarization_image,axis=0)
        self._number_of_zero_pixels=np.where(number_of_zero_pixels==0,1,number_of_zero_pixels)

    #取得した散乱パターン関数から学習データと教師データを作成(シングルプロセス)
    def _make_train_and_test_data_singleprocess(self,id):
        #鉛直方向の成分(散乱パターン関数)(鉛直方向に必ず原子が存在)
        if self.only_learning_existence_atom:
            print("Existence of atoms in the vertical direction")
            print("Add up the scattering pattern functions(Vertical direction)")
            out_data,out_label=self._sum_scattering_pattern_function_vertical_component_atom_existence(id)
            if self.angular_direction:
                #角度方向の成分(散乱パターン関数)
                print("Add up the scattering pattern functions(Angular direction)")
                out_data_angle,_=self._sum_scattering_pattern_function_angle_component_singleprocess(id)
                out_data+=out_data_angle
        else:
            #鉛直方向の成分(散乱パターン関数)(鉛直方向に必ず原子が存在しない)
            atoms_existence_nonexistence=random.randint(0,3)
            if atoms_existence_nonexistence==0:
                print("Non-existence of atoms in the vertical direction")
                print("Add up the scattering pattern functions(Angular direction)")
                out_data,out_label=self._sum_scattering_pattern_function_angle_component_singleprocess(id)
            else:
                print("Existence of atoms in the vertical direction")
                print("Add up the scattering pattern functions(Vertical direction)")
                out_data,out_label=self._sum_scattering_pattern_function_vertical_component_atom_existence(id)
                if self.angular_direction:
                    #角度方向の成分(散乱パターン関数)
                    print("Add up the scattering pattern functions(Angular direction)")
                    out_data_angle,_=self._sum_scattering_pattern_function_angle_component_singleprocess(id)
                    out_data+=out_data_angle
        out_data_numpy=np.array([out_data],dtype=np.float32)
        out_label_numpy=np.array(out_label,dtype=np.float32)
        out_data_tensor=self._transform(out_data_numpy)
        out_label_tensor=torch.from_numpy(out_label_numpy).clone()
        return out_data_tensor,out_label_tensor

    #取得した散乱パターン関数から学習データと教師データを作成(マルチプロセス)
    def _make_train_and_test_data_multiprocess(self,id):
        #鉛直方向の成分(散乱パターン関数)(鉛直方向に必ず原子が存在)
        if self.only_learning_existence_atom:
            print("Existence of atoms in the vertical direction")
            print("Add up the scattering pattern functions(Vertical direction)")
            out_data,out_label=self._sum_scattering_pattern_function_vertical_component_atom_existence(id)
            if self.angular_direction:
                #角度方向の成分(散乱パターン関数)
                print("Add up the scattering pattern functions(Angular direction)")
                out_data_angle,_=self._sum_scattering_pattern_function_angle_component_multiprocess(id)
                out_data+=out_data_angle
        else:
            #鉛直方向の成分(散乱パターン関数)(鉛直方向に必ず原子が存在しない)
            atoms_existence_nonexistence=random.randint(0,3)
            if atoms_existence_nonexistence==0:
                print("Non-existence of atoms in the vertical direction")
                print("Add up the scattering pattern functions(Angular direction)")
                out_data,out_label=self._sum_scattering_pattern_function_angle_component_multiprocess(id)
            else:
                print("Existence of atoms in the vertical direction")
                print("Add up the scattering pattern functions(Vertical direction)")
                out_data,out_label=self._sum_scattering_pattern_function_vertical_component_atom_existence(id)
                if self.angular_direction:
                    #角度方向の成分(散乱パターン関数)
                    print("Add up the scattering pattern functions(Angular direction)")
                    out_data_angle,_=self._sum_scattering_pattern_function_angle_component_multiprocess(id)
                    out_data+=out_data_angle
        out_data_numpy=np.array([out_data],dtype=np.float32)
        out_label_numpy=np.array(out_label,dtype=np.float32)
        out_data_tensor=self._transform(out_data_numpy)
        out_label_tensor=torch.from_numpy(out_label_numpy).clone()
        return out_data_tensor,out_label_tensor

    #鉛直方向の成分の散乱パターンの足し合わせ(鉛直方向に原子が存在)
    def _sum_scattering_pattern_function_vertical_component_atom_existence(self,id):
        out_data=np.zeros(len(self._sp_image[0]))
        out_label=np.zeros(len(self._atomic_distance))
        random_number_vertical_array=np.random.randint(0,len(self._sp_image)-1,id)
        random_number_vertical_array_unique=np.unique(random_number_vertical_array)
        #鉛直方向の成分(散乱パターン関数)
        for i in random_number_vertical_array_unique:
            out_data+=self._sp_image[i]
            out_label[i]+=1
        return out_data,out_label

    #角度方向の成分の散乱パターンの足し合わせ(シングルプロセス)
    def _sum_scattering_pattern_function_angle_component_singleprocess(self,id):
        out_data_angle=np.zeros(len(self._sp_image[0]))
        out_label=np.zeros(len(self._atomic_distance))
        #角度方向の成分(散乱パターン関数)
        random_number_angle_sp_array=np.random.randint(0,len(self._sp_image)-1,id)
        random_number_angle_sp_array_unique=np.unique(random_number_angle_sp_array)
        random_number_angle_array=np.random.randint(0,self.angle_max,id)
        random_number_angle_array_unique=np.unique(random_number_angle_array)
        if len(random_number_angle_sp_array_unique)<len(random_number_angle_array_unique):
            random_number_angle_array_unique=np.delete(random_number_angle_array_unique,np.s_[len(random_number_angle_sp_array_unique):])
        elif len(random_number_angle_sp_array_unique)>len(random_number_angle_array_unique):
            random_number_angle_sp_array_unique=np.delete(random_number_angle_sp_array_unique,np.s_[len(random_number_angle_array_unique):])
        else:
            pass
        for i in range(len(random_number_angle_sp_array_unique)):
            print("Converting from scattering pattern function to orthographic projection")
            orthographic=self._convert_from_scattering_pattern_to_orthographic(random_number_angle_sp_array_unique[i])
            print("image is rotating")
            print(f'Rotation angle={random_number_angle_array_unique[i]}°')
            rotation_image=self._rotation_image_PIL(orthographic,random_number_angle_array_unique[i])
            print("circular integral")
            circular_mean=self._circular_integral_orthographic(rotation_image)
            out_data_angle+=circular_mean
        return out_data_angle,out_label

    #角度方向の成分の散乱パターンの足し合わせ(マルチプロセス)
    def _sum_scattering_pattern_function_angle_component_multiprocess(self,id):
        out_data_angle=np.zeros(len(self._sp_image[0]))
        out_label=np.zeros(self._atomic_distance)
        #角度方向の成分(散乱パターン関数)
        random_number_angle_sp_array=np.random.randint(0,len(self._sp_image)-1,id)
        random_number_angle_sp_array_unique=np.unique(random_number_angle_sp_array)
        random_number_angle_array=np.random.randint(0,self.angle_max,id)
        random_number_angle_array_unique=np.unique(random_number_angle_array)
        if len(random_number_angle_sp_array_unique)<len(random_number_angle_array_unique):
            random_number_angle_array_unique=np.delete(random_number_angle_array_unique,np.s_[len(random_number_angle_sp_array_unique):])
        elif len(random_number_angle_sp_array_unique)>len(random_number_angle_array_unique):
            random_number_angle_sp_array_unique=np.delete(random_number_angle_sp_array_unique,np.s_[len(random_number_angle_array_unique):])
        else:
            pass
        circular_mean=self._wrapper_image_process_multiprocess(random_number_angle_sp_array_unique,random_number_angle_array_unique)
        circular_mean=np.sum(circular_mean,axis=0)
        out_data_angle+=circular_mean
        return out_data_angle,out_label

    #マルチプロセス、散乱パターン関数を正射図法に変換、画像回転、円環積分
    def _wrapper_image_process_multiprocess(self,index,angle):
        print("Converting from scattering pattern function to orthographic projection")
        print("image is rotating")
        print(f'Rotation angle={angle}°')
        print("circular integral")
        with concurrent.futures.ProcessPoolExecutor(max_workers=self._number_cpu) as executor:
            result_c=np.array(list(executor.map(self._convert_from_scattering_pattern_to_orthographic,index)))
            result_r=np.array(list(executor.map(self._rotation_image_PIL,result_c,angle)))
            result_i=np.array(list(executor.map(self._circular_integral_orthographic,result_r)))
        return result_i

    #散乱パターン関数を正射図法に変換
    def _convert_from_scattering_pattern_to_orthographic(self,index):
        orthographic_tile=np.tile(self._sp_image[index],[len(self._orthographic),1])
        orthographic=orthographic_tile*self._binarization_image
        return orthographic

    #画像回転操作(Pillow使用)
    #バイリニアー補間
    def _rotation_image_PIL(self,image,angle):
        pil_image=Image.fromarray(image)
        rotated_image=np.array(pil_image.rotate(angle,resample=Image.BILINEAR,expand=0))
        return rotated_image

    #正射図法の円環積分(散乱パターン関数)
    def _circular_integral_orthographic(self,image):
        circular_sum=np.sum(image,axis=0)
        circular_mean=circular_sum/self._number_of_zero_pixels
        return circular_mean

if __name__ == "__main__":
    dataset_NN=Dataset_from_AIR_NN()
    dataset_NN()