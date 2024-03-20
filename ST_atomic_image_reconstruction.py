"Copyright (C) S.Takeuchi 2024 All Rights Reserved."

import numpy as np
import api_3d_air_image
from PIL import Image
import torch
import quaternion
import os
import concurrent.futures
from matplotlib import pyplot as plt

"Atomic Image Reconstruction-Neural Network(AIR-NN)"
#インスタンス作製
api=api_3d_air_image.Api()
api.connect()

#ニューラルネットワークで学習した散乱パターン関数から原子像を再構成するクラス
class atomic_image_reconstruction:
    def __init__(self,x=float(10),y=float(10),z_min=float(0),z_max=float(10),step=float(0.1),r_min=float(1.5)):
        self._x_min_value=-x
        self._x_max_value=x
        self._y_min_value=-y
        self._y_max_value=y
        self._z_min_value=z_min
        self._z_max_value=z_max
        self._step_voxel=step
        self._non_reconstruction_range=int((z_min*-1+r_min)/step)
        self.chose_multi_process=False
        self.use_moveing_average=False
        self.convlute_range=5
        self._number_cpu=os.cpu_count()

    #ボクセル関数を作成
    def make_voxel(self):
        self._define_mesh_for_voxel()
        self._voxel_function=np.zeros([self._z_mesh_limit,self._x_mesh_limit,self._y_mesh_limit])
        #四元数で回転させる座標(基準座標)
        self._vertical_coordinates_of_voxel=np.zeros([self._z_mesh_limit,3])
        for i in range(self._z_mesh_limit):
            self._vertical_coordinates_of_voxel[i]=[0,0,i]

    #ボクセル関数のメッシュ計算,中心座標を取得
    def _define_mesh_for_voxel(self):
        x_min_mesh=int(self._x_min_value/self._step_voxel)
        x_max_mesh=int(self._x_max_value/self._step_voxel)
        self._x_mesh_limit=int(x_max_mesh-x_min_mesh+1)
        y_min_mesh=int(self._y_min_value/self._step_voxel)
        y_max_mesh=int(self._y_max_value/self._step_voxel)
        self._y_mesh_limit=int(y_max_mesh-y_min_mesh+1)
        z_min_mesh=int(self._z_min_value/self._step_voxel)
        z_max_mesh=int(self._z_max_value/self._step_voxel)
        self._z_mesh_limit=int(z_max_mesh-z_min_mesh+1)
        x_center=int(self._x_mesh_limit/2)
        y_center=int(self._y_mesh_limit/2)
        z_center=int(0-z_min_mesh)
        self._voxel_center=np.array([x_center,y_center,z_center])

    #ホログラムを取得(正距方位図法)
    def get_hologram_azimuthal(self):
        #画像の選択
        print("Please chose hologram image(Azimuthal projection)")
        Enter=str(input("適当な数値を入力="))
        wid=api.window_get_focused_wid()
        self._image=api.image_get(wid)
        #self.__input_data=np.zeros(len(self.__image))
        image_caption=api.window_caption_get(wid)
        print(f'caption of hologram image:{image_caption}')
        self._convert_from_azimuthal_map_to_cartesian_coordinates_of_sphere()
        return self._image

    #原子像再生に必要なデータをclass内に読み込み
    def get_data_for_atomic_image_reconstruction(self,model,cal_range_x,cal_range_y):
        self._model=model
        index_x=[]
        index_y=[]
        center_vec=np.array([self._center[0],self._center[1]])
        for i in range(cal_range_x[0],cal_range_x[1],cal_range_x[2]):
            for j in range(cal_range_y[0],cal_range_y[1],cal_range_y[2]):
                vec=np.array([i,j])
                k=np.linalg.norm(vec-center_vec)
                if k<=self._center[0]:
                    index_x.append(i)
                    index_y.append(j)
        self._index_x=index_x
        self._index_y=index_y

    #原子像再生、マルチプロセスかシングルプロセスを選択
    def atomic_image_reconstruction(self,threshold=0):
        if self.chose_multi_process:
            self._atomic_image_reconstructuion_multiprocess(threshold=threshold)
        else:
            self._atomic_image_reconstructuion_singleprocess(threshold=threshold)

    #各ピクセルごとに原子像再生(シングルプロセス)
    def _atomic_image_reconstructuion_singleprocess(self,threshold=0):
        for x,y in zip(self._index_x,self._index_y):
            line=self._circular_integral_direct_from_Azimuthal_map_ver1(x,y)
            line_tensor=self._change_to_tensor_form(line)
            output=self._model(line_tensor)#推論用の入力dataをinputし、出力を求める
            atom_position=output.to('cpu').detach().numpy().copy()
            self._input_reconstruction_result_into_voxel_function(x,y,atom_position,threshold)

    #マルチプロセスのために円環積分、四元数を使用した座標変換、再構成結果の関数をラッピング
    def _wrapper_function_for_multiprocess(self,x,y):
        atom_position=[]
        line=self._circular_integral_direct_from_Azimuthal_map_ver1(x,y)
        if self.use_moveing_average:
            line=self._moveing_average(line,conv_range=self.convlute_range)
            print("Moveing average. . .")
        line_tensor=self._change_to_tensor_form(line)
        output=self._model(line_tensor)#推論用の入力dataをinputし、出力を求める
        atom_position.append(output.to('cpu').detach().numpy().copy())
        atom_position=np.array(atom_position)
        return atom_position

    #各ピクセルごとの原子像再生を並列化(マルチプロセス)
    def _atomic_image_reconstructuion_multiprocess(self,threshold=0):
        #print(f'circular integral(x,y):{self.index_x,self.index_y}')
        #print(f'convert from reconstruction result(x,y):{self.index_x,self.index_y}')
        #print("Reconstruction result, inputting. . .")
        with concurrent.futures.ProcessPoolExecutor(max_workers=self._number_cpu) as executor:
            atom_position=np.array(list(executor.map(self._wrapper_function_for_multiprocess,self._index_x,self._index_y)))
        for x,y,i in zip(self._index_x,self._index_y,range(len(atom_position))):
            self._input_reconstruction_result_into_voxel_function(x,y,atom_position[i][0],threshold)

    #Azimuthal mapから球面の直交座標に変換
    def _convert_from_azimuthal_map_to_cartesian_coordinates_of_sphere(self):
        Cx=int((len(self._image)+1)/2)
        Cy=int((len(self._image[0])+1)/2)
        self._resolution=90/Cx
        self._center=np.array([Cx,Cy])
        #内積の参照値
        self._inner_product_reference=np.sort(np.cos(np.arange(0,np.radians(90),np.radians(self._resolution))))
        self._uxi=np.zeros([len(self._image),len(self._image[0])])
        self._uyi=np.zeros([len(self._image),len(self._image[0])])
        self._uzi=np.zeros([len(self._image),len(self._image[0])])
        for i in range(len(self._image)):
            for j in range(len(self._image[0])):
                theta=np.radians(np.sqrt((i-Cx)**2+(j-Cy)**2)*self._resolution)
                phi=np.arctan2(j-Cy,i-Cx)
                self._uxi[i][j]=np.sin(theta)*np.cos(phi)
                self._uyi[i][j]=np.sin(theta)*np.sin(phi)
                self._uzi[i][j]=np.cos(theta)
        uxi_ravel=np.ravel(self._uxi)
        uyi_ravel=np.ravel(self._uyi)
        uzi_ravel=np.ravel(self._uzi)
        u_xyzi_vec=np.stack([uxi_ravel,uyi_ravel,uzi_ravel],axis=1)
        u_xyzi_vec_norm=np.linalg.norm(u_xyzi_vec,axis=1)
        u_xyzi_vec_norm_stack=np.stack([u_xyzi_vec_norm,u_xyzi_vec_norm,u_xyzi_vec_norm],axis=1)
        u_xyzi_vec_normal=u_xyzi_vec/u_xyzi_vec_norm_stack
        #単位ベクトルに変換(原点から球面上の点)
        self.__uxi_normal=np.reshape(u_xyzi_vec_normal[:,0],[len(self._image),len(self._image[0])])
        self.__uyi_normal=np.reshape(u_xyzi_vec_normal[:,1],[len(self._image),len(self._image[0])])
        self.__uzi_normal=np.reshape(u_xyzi_vec_normal[:,2],[len(self._image),len(self._image[0])])
        #単位ベクトルマップの中心から端までを切り取り
        self.__cartesian_coordinate_crop=np.zeros([3,self._center[0]])
        self.__cartesian_coordinate_crop[0]=self.__uxi_normal[self._center[0],:self._center[1]]
        self.__cartesian_coordinate_crop[1]=self.__uyi_normal[self._center[0],:self._center[1]]
        self.__cartesian_coordinate_crop[2]=self.__uzi_normal[self._center[0],:self._center[1]]

    #Azimuthal mapから直接、円環積分した結果は正射図法ver1(x,yはピクセルの座標)
    def _circular_integral_direct_from_Azimuthal_map_ver1(self,x,y):
        theta=np.radians(np.sqrt((x-self._center[0])**2+(y-self._center[1])**2)*self._resolution)
        phi=np.arctan2(y-self._center[1],x-self._center[0])
        print(f'circular integral(x,y):{x,y}')
        center_vec=np.array([np.sin(theta)*np.cos(phi),
                            np.sin(theta)*np.sin(phi),
                            np.cos(theta)])
        center_vec_normal=center_vec/np.linalg.norm(center_vec)
        inner_product_x_component=self.__uxi_normal*center_vec_normal[0]
        inner_product_y_component=self.__uyi_normal*center_vec_normal[1]
        inner_product_z_component=self.__uzi_normal*center_vec_normal[2]
        inner_product_map=np.round(inner_product_x_component+inner_product_y_component+inner_product_z_component,3)
        inner_product_map_ravel=np.ravel(inner_product_map)
        #0以下の値を0に修正
        inner_product_map_where=np.where(inner_product_map_ravel<0.01,0.01,inner_product_map_ravel)
        inner_product_map_unique=np.unique(inner_product_map_where)
        circular_mean_average=np.zeros(len(inner_product_map_unique))
        weight=np.zeros(len(inner_product_map_unique))
        for i in range(len(inner_product_map_unique)):
            index=np.where(inner_product_map==inner_product_map_unique[i])
            for j in range(len(index[0])):
                if self._image[index[0][j]][index[1][j]]!=0:
                    circular_mean_average[i]+=self._image[index[0][j]][index[1][j]]
                    weight[i]+=1
        circular_mean_average=circular_mean_average/weight
        circular_mean_average_pil=Image.fromarray(circular_mean_average)
        circular_mean_average_pil_resize=circular_mean_average_pil.resize((1,self._center[0]),resample=Image.BILINEAR)
        circular_mean_average_numpy=np.array(circular_mean_average_pil_resize)
        circular_mean_average_numpy=circular_mean_average_numpy[0:,0]
        return circular_mean_average_numpy

    #tensor形に変更
    def _change_to_tensor_form(self,line):
        input_data_normalization=self._max_min_normalization(line)
        line_tensor=torch.from_numpy(input_data_normalization.astype(np.float32)).clone()
        return line_tensor

    #四元数を使用した座標変換(x,yはピクセルの座標)
    def _coordinate_transformation_using_quaternion(self,x,y):
        if self._center[0]==x and self._center[1]==y:
            pass
        else:
            theta=np.radians(np.sqrt((x-self._center[0])**2+(y-self._center[1])**2)*self._resolution)
            phi=np.arctan2(y-self._center[1],x-self._center[0])
            #単位ベクトル
            unit_vector_z=np.array([0,0,1])
            #回転させたい軸の方向ベクトル
            Vxyz=np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
            #回転角度
            rotation_angel=np.arccos(np.sum(unit_vector_z*Vxyz)/(np.linalg.norm(Vxyz)))
            #回転させる軸ベクトル
            Rvec=np.array([[0,-1],[1,0]])
            xy=np.array([Vxyz[0],Vxyz[1]])
            rxy=np.dot(xy,Rvec)
            Rxyz=np.append(rxy,0)
            w=np.cos(rotation_angel/2)
            distance=np.linalg.norm(Rxyz)
            Rxyz_normal=(Rxyz/distance)*np.sin(rotation_angel/2)
            #クォータニオンで回転
            self.__quat=np.quaternion(w,Rxyz_normal[0],Rxyz_normal[1],Rxyz_normal[2])
            self.__quat_conjugate=np.quaternion(w,-Rxyz_normal[0],-Rxyz_normal[1],-Rxyz_normal[2])

    #再構成結果をボクセル関数の座標に変換
    def _input_reconstruction_result_into_voxel_function(self,x,y,result_array,threshold):
        print(f'convert from reconstruction result(x,y):{x,y}')
        print("Reconstruction result, inputting. . .")
        #np.put(result_array,range(0,self.__non_reconstruction_range),0)#散乱パターン関数のない範囲を0にする
        additional_voxel=np.zeros(self._non_reconstruction_range)
        result_array=np.append(additional_voxel,result_array)
        result_max=np.max(result_array)
        value=result_max*threshold
        result_array=np.where(result_array>value,result_array,0)
        self._coordinate_transformation_using_quaternion(x,y)
        if self._center[0]==x and self._center[1]==y:
            for i in range(len(result_array)):
                x_reconstruction=int(self._vertical_coordinates_of_voxel[i][0]+self._voxel_center[0])
                y_reconstruction=int(self._vertical_coordinates_of_voxel[i][1]+self._voxel_center[1])
                z_reconstruction=int(self._vertical_coordinates_of_voxel[i][2])
                #ボクセル関数に座標を変換しながら再構成結果を入力(x,yはピクセルの座標)
                if 0<=x_reconstruction<self._x_mesh_limit and 0<=y_reconstruction<self._y_mesh_limit and 0<=z_reconstruction<self._z_mesh_limit:
                    self._voxel_function[z_reconstruction][x_reconstruction][y_reconstruction]=result_array[i]
        else:
            for i in range(len(result_array)):
                U_vec=np.quaternion(self._vertical_coordinates_of_voxel[i][0],
                                      self._vertical_coordinates_of_voxel[i][1],
                                      self._vertical_coordinates_of_voxel[i][2])
                coordinate=np.imag(self.__quat_conjugate*U_vec*self.__quat)
                x_reconstruction=int(coordinate[0]+self._voxel_center[0])
                y_reconstruction=int(coordinate[1]+self._voxel_center[1])
                z_reconstruction=int(coordinate[2])
            #ボクセル関数に座標を変換しながら再構成結果を入力(x,yはピクセルの座標)
                if 0<=x_reconstruction<self._x_mesh_limit and 0<=y_reconstruction<self._y_mesh_limit and 0<=z_reconstruction<self._z_mesh_limit:
                    self._voxel_function[z_reconstruction][x_reconstruction][y_reconstruction]=result_array[i]

    #最大値最小値でデータの正規化
    def _max_min_normalization(self,data):
        data_max=np.max(data)
        data_min=np.min(data)
        if data_max==data_min:
            data_normalization=data
        else:
            data_normalization=(data-data_min)/(data_max-data_min)
        data_normalization_array=data_normalization
        return data_normalization_array

    #移動平均
    def _moveing_average(self,list_array,conv_range=15):
        return np.convolve(list_array,np.ones(conv_range),mode='same')/conv_range

    #volumeimageを作成して3Dairに出力
    def convert_to_volume_image_on_3Dair(self,threshold=0,soft_focus=1):
        max_value=np.max(self._voxel_function)
        threshold_value=max_value*threshold
        voxel_change_value=np.where(self._voxel_function>threshold_value,1,0)
        voxel_soft_focus=self._soft_focus_on_volumedata(voxel_change_value,soft_focus=soft_focus)
        #強度値をそろえる
        voxel_change_value_soft_focus=np.where(voxel_soft_focus>1,1,0)
        image3D_wid=api.image3D_new_window()
        api.image3D_set(image3D_wid,voxel_change_value_soft_focus,
        self._x_min_value,self._x_max_value,self._y_min_value,self._y_max_value,self._z_min_value,self._z_max_value)
        print("Output volume image")

    def _soft_focus_on_volumedata(self,volume_data,soft_focus=0):
        add_array=np.zeros([soft_focus*5,self._x_mesh_limit,self._y_mesh_limit])#z軸方向に余分な配列を追加(虚像削除のため)
        volume_data=np.append(volume_data,add_array,axis=0)
        volume_soft_focus=volume_data
        if soft_focus!=0:#voxel関数をずらしながら足している(ソフトフォーカス)
            for i in range(-soft_focus,soft_focus+1,1):
                volume_soft_focus+=np.roll(volume_soft_focus,i,axis=1)
                volume_soft_focus+=np.roll(volume_soft_focus,i,axis=2)
                volume_soft_focus+=np.roll(volume_soft_focus,i,axis=0)
                volume_soft_focus+=np.roll(volume_soft_focus,(i,i),axis=(1,2))
                volume_soft_focus+=np.roll(volume_soft_focus,(-i,i),axis=(1,2))
            for j in range(-soft_focus,1,1):
                volume_soft_focus+=np.roll(volume_soft_focus,(j,j),axis=(0,2))
                volume_soft_focus+=np.roll(volume_soft_focus,(-j,j),axis=(0,2))
                volume_soft_focus+=np.roll(volume_soft_focus,(j,j),axis=(0,1))
                volume_soft_focus+=np.roll(volume_soft_focus,(-j,j),axis=(0,1))
            volume_soft_focus=np.delete(volume_soft_focus,
            np.s_[len(volume_soft_focus)-soft_focus*5:len(volume_soft_focus)],axis=0) #z軸方向にずらして足した際の虚像を消す
        return volume_soft_focus

if __name__ == "__main__":
    import time
    ar=atomic_image_reconstruction()
    ar.get_hologram_azimuthal()
    #ar.get_scattering_pattern_function()
    t1=time.time()
    circular_ave1=ar._circular_integral_direct_from_Azimuthal_map_ver1(180,180)
    #line=ar._convolution_of_scattering_pattern_and_hologram(circular_ave1)
    #circular_ave2=ar.circular_integral_azimuthal(180,180)
    t2=time.time()
    elapsed_time=(t2-t1)
    print(f'経過時間(minute){elapsed_time}秒')
    a=np.zeros(361)
    #circular_ave1=ar.change_to_tensor_form(circular_ave1,a)
    print(len(circular_ave1))
    plt.plot(range(len(circular_ave1)),circular_ave1)
    #plt.xlim(180,360)
    #plt.plot(range(len(circular_ave2)),circular_ave2)
    plt.show()