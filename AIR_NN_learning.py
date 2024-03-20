import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import ST_Neural_Network
import ST_Dataset
import time
from matplotlib import pyplot as plt

"Atomic Image Reconstruction-Neural Network(AIR-NN)"
if __name__=="__main__":#マルチプロセスを使用するために必要
    t1=time.time()
    ###パラメーター設定(開始)
    #学習モデルの保存名
    save_name="model_practice"
    #save_name="model_Si_R10_s_wave_step0.1_loop7000_vertical_component_angle0_angular_component_angle0_atom20_1"
    #学習の繰り返し数l
    EPOCH=100
    ###データセット作成
    #学習用Datasetのインスタンス作成
    #r_min(散乱パターン関数の最小距離)、r_max(散乱パターン関数の最大距離)
    #z_min(再構成する原子位置の最小値)、z_max(再構成する原子位置の最大値)、step(再構成する原子のステップ)
    dataset_train=ST_Dataset.Dataset_from_AIR_NN(r_min=1.5,r_max=10)
    #散乱パターン関数の使用するデータ範囲を設定(正距方位図法で360pixなら180にする)
    dataset_train.scattering_data_range_pix=180
    #鉛直方向に必ず原子がある場合で学習
    dataset_train.only_learning_existence_atom=True
    #シングルプロセスかマルチプロセスを選択(鉛直方向に必ず原子がある場合での学習ではFalse)
    dataset_train.chose_multi_process=False
    #鉛直以外の原子の散乱パターン関数と鉛直上の散乱パターン関数を足し合わせて学習させる場合(angle_maxの値が入る)
    dataset_train.angular_direction=False
    #0から任意の範囲の散乱パターンを0にして学習
    #dataset_train.set_arbitrary_range_to_zero=False
    #バッチサイズ
    Batch_size=10
    #配置する原子数
    dataset_train.number_of_atoms=10
    #影響を考慮する角度範囲
    dataset_train.angle_max=90
    #NNのインスタンス作製
    dataset_train.get_scattering_pattern_function()
    dataset_train.pre_calculate_parameters_for_various_calculations()
    NN=ST_Neural_Network.AirNN(nch=16,output_layer=dataset_train.output_layer,conv_kernel_size=(1,3),padding_size=(0,1),pool='max',pool_kernel_size=(1,2),batch_norm=True)
    #NN=ST_Neural_Network.NN_for_AIR_NN_parallel_2_ver1(input_layer,output_layer,learning_mode=True)
    ###パラメーター設定(終了)

    ###学習
    #計算deviceの自動決定,GPUがあれば使用、無ければCPUを使用
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #ニューラルネットワークを計算deviceに渡す
    model=NN.to(device)
    #誤差関数の決定(二乗誤差)
    criterion=nn.MSELoss()
    #最適化方法の決定(Adam)
    optimizer=optim.Adam(model.parameters(),lr=0.001,betas=(0.9,0.999),eps=1e-8,weight_decay=0,amsgrad=True)
    #学習用のDataLoaderを作成
    train_loader=DataLoader(dataset_train,batch_size=Batch_size,shuffle=True)
    #ネットワークを訓練モードに切り替える
    model.train()
    for epoch in range(EPOCH):
        print(f'epoch{epoch+1}')
        for sp,line in train_loader:#データセットの学習データと教師データを渡す
            sp,line=sp.to(device),line.to(device)#学習データと教師データを計算deviseに渡したあと変数に代入
            optimizer.zero_grad()#勾配を初期化(0に戻す)
            output_train=model(sp)#順伝播計算
            #print("学習(順伝播)=",output_train)
            loss=criterion(output_train,line)#誤差計算
            plt.plot(range(len(line[0])),line[0].detach().numpy())
            plt.plot(range(len(line[0])),output_train[0].detach().numpy())
            plt.show()
            print(f'学習(誤差)={loss}')
            loss.backward()#誤差逆伝播計算
            del loss#余計な計算グラフを削除
            optimizer.step()#パラメーターを更新
    ###
    #モデルの保存
    model_path=f'{save_name}.pth'
    torch.save(model.to('cpu').state_dict(),model_path)

    t2=time.time()
    elapsed_time=(t2-t1)/3600
    print(f'経過時間(minute){elapsed_time}時間')
    #elapsed_time=(t2-t1)/3600
    #print("経過時間(hour)",elapsed_time,"時間")
