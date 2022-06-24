FROM nvidia/cuda:11.7.0-runtime-ubuntu20.04

# Set frontend as non-interactive
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update

# Install python and pip
RUN apt-get install -y python3-opencv python3-pip && \
    python3 -m pip install --upgrade pip && \
    apt-get -y install git

# Install nebullvm
ARG NEBULLVM_VERSION=latest
RUN if [ "$NEBULLVM_VERSION" = "latest" ] ; then \
        pip install nebullvm ; \
    else \
        pip install nebullvm==${NEBULLVM_VERSION} ; \
    fi

# Install required python modules
RUN pip install scipy==1.5.4 && \
    pip install cmake

# Install default deep learning compilers
ARG COMPILER=all
RUN if [ "$COMPILER" = "all" ] ; then \
        python3 -c "import nebullvm" ; \
    elif [ "$COMPILER" = "tensorrt" ] ; then \
        python3 -c "from nebullvm.installers.installers import install_tensor_rt; install_tensor_rt()" ; \
    elif [ "$COMPILER" = "openvino" ] ; then \
        python3 -c "from nebullvm.installers.installers import install_openvino; install_openvino()" ; \
    elif [ "$COMPILER" = "onnxruntime" ] ; then \
        python3 -c "from nebullvm.installers.installers import install_onnxruntime; install_onnxruntime()" ; \
    fi

# Install TVM
RUN if [ "$COMPILER" = "all" ] || [ "$COMPILER" = "tvm" ] ; then \
        python3 -c "from nebullvm.installers.installers import install_tvm; install_tvm()" ; \
        #### Trick to fix tvm configs issue
        mv /root/tvm/configs /root/tvm/configs_orig ; touch /root/tvm/configs ; \
        python3 -c "from nebullvm.installers.installers import install_tvm; install_tvm()" ; \
        # Manually copy the configs folder to the correct path
        export TVM_DIR_NAME=$(ls -d /root/.local/lib/python3.8/site-packages/tvm*) ; cp -R /root/tvm/configs_orig "$TVM_DIR_NAME/tvm/configs" ; \
        # RUN /root/.local/bin/tvmc
        ####
        python3 -c "from tvm.runtime import Module" ; \
    fi