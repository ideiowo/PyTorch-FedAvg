# PyTorch-FedAvg
Simple FedAvg with PyTorch

FedAVG/
│
├── data/                 # 數據集存放位置
│   └── CIFAR10/
|        ├──train
|        ├──validation
|        ├──trainLabels.csv
│
├── client/
│   ├── __init__.py
│   ├── client.py      # 客戶端訓練和數據處理
│
├── server/
│   ├── __init__.py
│   ├── server.py      # 伺服器端模型聚合和更新(梯度平均)
│
├── models/
│   ├── __init__.py
│   ├── architecture.py  # 定義模型的架構(DNN、CNN、ResNet)
│
├── utils/
│   ├── __init__.py
│   ├── data_utils.py  # 數據處理和加載的工具
│   ├── move_png.py　　#將kaggle照片分類成資料夾
│
└── main.py            # 主執行檔案，組織整個FedAVG流程

