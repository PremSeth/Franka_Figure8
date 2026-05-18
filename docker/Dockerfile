# Stable Isaac Lab 2.3 prebuilt container. Override at build time if needed:
#   docker build --build-arg ISAAC_LAB_IMAGE=nvcr.io/nvidia/isaac-lab:3.0.0-beta1 ...
ARG ISAAC_LAB_IMAGE=nvcr.io/nvidia/isaac-lab:2.3.0
FROM ${ISAAC_LAB_IMAGE}

ENV ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    PYTHONUNBUFFERED=1

# Some Isaac Lab containers expose Python primarily through /isaac-sim/python.sh.
# Make the README's `python ...` commands work consistently in non-interactive Docker runs.
RUN if ! command -v python >/dev/null 2>&1; then ln -s /isaac-sim/python.sh /usr/local/bin/python; fi

WORKDIR /workspace/Franka_Figure8
COPY . .

# Install this external Isaac Lab project into the container's Isaac Lab Python environment.
RUN python -m pip install --no-cache-dir -e source/Franka_End_Effector_Tracking \
    && python -c "import isaaclab; import isaacsim; import isaaclab_rl; import rsl_rl; print('Isaac Lab Docker import check OK')"

CMD ["bash"]
