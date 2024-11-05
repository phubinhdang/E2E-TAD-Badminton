FROM tad_badminton_base
WORKDIR /tad
COPY . .
RUN cd models/ops && python setup.py build_ext --inplace & cd ../..
CMD ["python", "main.py", "--cfg", "configs/badminton_e2e_slowfast_tadtr.yml", "--multi_gpu", "--num_workers", "12"]
