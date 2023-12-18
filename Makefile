.PHONY: all detection estimation reconfigure

# option for debug build from command line
BUILD_TYPE ?= Release

define build_target
	ninja -C src/$(1)/build
endef

define configure_target
	mkdir -p src/$(1)/build
	cd src/$(1)/build && cmake -G Ninja .. -D CMAKE_BUILD_TYPE=$(BUILD_TYPE)
endef

all: detection estimation

detection:
	$(call build_target,detection)

estimation:
	$(call build_target,estimation)

configure:
	$(call configure_target,detection)
	$(call configure_target,estimation)

clean:
	rm -rf src/detection/build
	rm -rf src/estimation/build

test:
	$(call build_target,estimation)
	./src/estimation/build/flow_velocity_test