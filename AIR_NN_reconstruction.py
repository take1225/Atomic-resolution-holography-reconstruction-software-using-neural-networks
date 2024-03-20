import torch
import ST_Neural_Network
import ST_atomic_image_reconstruction
import time

"Atomic Image Reconstruction-Neural Network(AIR-NN)"
#保存モデルの読み込み
load_model_name="model_practice"
#load_model_name="model_Si_R10_s_wave_step0.1_loop7000_vertical_component_angle0_angular_component_angle0_atom20_1"

#インスタンス作成
#x、y(再構成するx軸とy軸の範囲)、z_min(再構成する原子位置の最小値)z_max(再構成する原子位置の最大値)step(再構成する原子のステップ)
#r_min(散乱パターン関数の最小距離)
rf=ST_atomic_image_reconstruction.atomic_image_reconstruction(x=10,y=10,z_min=-0.5,z_max=10,step=0.1,r_min=1.5)
rf.make_voxel()
#入力層と出力層の設定(学習時のデータを使用)
input_layer=181
output_layer=86
#計算範囲の指定(ピクセル値)
x_start_stop_step=(0,380,20)
y_start_stop_step=(0,380,20)
#シングルプロセスかマルチプロセスを選択
rf.chose_multi_process=True
#移動平均(円環平均に対して)
rf.use_moveing_average=False
rf.convlute_range=5
device='cuda' if torch.cuda.is_available() else 'cpu'
#新しいモデル
model=ST_Neural_Network.NN_for_AIR_NN_ver1(input_layer,output_layer).to(device)
#model=ST_Neural_Network.NN_for_AIR_NN_parallel_2_ver1(input_layer,output_layer,learning_mode=False).to(device)

#保存したモデルパラメータの読み込み
model.load_state_dict(torch.load(f'{load_model_name}.pth'))

if __name__ == "__main__":#マルチプロセスを使用するために必要
    ###推論(1枚のホログラム)
    #ネットワークを推論モードに切り替える
    model.eval()
    rf.get_hologram_azimuthal()
    rf.get_data_for_atomic_image_reconstruction(model,x_start_stop_step,y_start_stop_step)
    t1=time.time()
    rf.atomic_image_reconstruction(threshold=0.5)
    rf.convert_to_volume_image_on_3Dair(threshold=0.3,soft_focus=1)
    del model
    del rf
    t2=time.time()
    elapsed_time=(t2-t1)/3600
    print(f'経過時間(minute){elapsed_time}時間')