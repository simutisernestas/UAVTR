# export container
FROM pytorch/pytorch

RUN pip3 install --upgrade pip
RUN pip3 install -U --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
RUN pip3 install onnx onnxruntime

CMD ["/bin/bash"]

# run
# docker build -t pytorch/nightly .
# docker run -it -v .:/workspace pytorch/nightly