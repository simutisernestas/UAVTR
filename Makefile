.PHONY: all detection estimation reconfigure

# option for debug build from command line
BUILD_TYPE ?= Release

define build_target
	ninja -C src/$(1)/build
endef

# -D CMAKE_CXX_FLAGS="-DEIGEN_NO_DEBUG=1"
define configure_target
	mkdir -p src/$(1)/build
	cd src/$(1)/build && cmake -G Ninja .. -D CMAKE_BUILD_TYPE=$(BUILD_TYPE)
endef

all: detection fusion estimation

detection:
	$(call build_target,detection)

fusion:
	$(call build_target,Fusion)

estimation: fusion
	$(call build_target,estimation)

configure:
	wget https://github.com/microsoft/onnxruntime/releases/download/v1.15.1/onnxruntime-training-linux-x64-1.15.1.tgz
	tar -xf onnxruntime-training-linux-x64-1.15.1.tgz
	mv onnxruntime-training-linux-x64-1.15.1 src/detection/onnxruntime
	rm onnxruntime-training-linux-x64-1.15.1.tgz
	$(call configure_target,detection)
	git clone https://github.com/xioTechnologies/Fusion --branch=main src/Fusion
	$(call configure_target,Fusion)
	$(call configure_target,estimation)

clean:
	rm -rf src/detection/build
	rm -rf src/estimation/build
	rm -rf src/Fusion/build

test:
	$(call build_target,estimation)
	./src/estimation/build/flow_velocity_test