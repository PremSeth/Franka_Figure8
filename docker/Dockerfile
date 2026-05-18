ARG ISAAC_LAB_IMAGE=nvcr.io/nvidia/isaac-lab:3.0.0-beta1
FROM ${ISAAC_LAB_IMAGE}

WORKDIR /workspace/Franka_End_Effector_Tracking
COPY . .

RUN python -m pip install -e source/Franka_End_Effector_Tracking

ENV ACCEPT_EULA=Y
ENV PRIVACY_CONSENT=Y

CMD ["bash"]
