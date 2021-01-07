#！/bin/bash

# xsub

python main.py rec_ensemble \
           --config config/mv-ignet/ntu-xsub/test_hpgnet_ensemble_simple.yaml \
           --weights weights/xsub/xsub_HPGNet_epoch120_model.pt \
           --weights2 weights/xsub/xsub_HPGNet-complement_epoch120_model.pt

# xview

# python main.py rec_ensemble \
#            --config config/mv-ignet/ntu-xview/test_hpgnet_ensemble_simple.yaml \
#            --weights weights/xview/xview_HPGNet_epoch120_model.pt\
#            --weights2 weights/xview/xview_HPGNet-complement_epoch120_model.pt
