FROM tad_badminton_base
WORKDIR /tad
COPY . .
# build the cuda extension first, then run the training script
CMD ["bash", "-c", "cd models/ops && python setup.py build_ext --inplace && cd ../.. && python main.py --cfg configs/badminton_e2e_slowfast_tadtr.yml --multi_gpu --num_workers 12"]
