# Fonts Generation for Indic Scripts

Font style transfer for Indian scripts using GANs.  
Currently, supports only Devanagari.

# Code Setup

## Training

TODO

## Inference

TODO

## Web UI

Ensure you have installed StreamLit by `pip install streamlit`

To run: `streamlit run font_GAN_ui.py`

## Hosting UI on GCP

**Pre-requisites**
- Create a GCP VM (Compute Engine) with HTTP/HTTPS exposed
- Optional: [Set a static IP to the VM](https://cloud.google.com/compute/docs/ip-addresses/reserve-static-external-ip-address)
- Install a remote Terminal session mananger: `sudo apt install byobu`

**Running StreamLit server**
- Type `byobu` to open a session
- `cd` into the project directory
- Run UI on port 80: `sudo env PATH=$PATH streamlit run font_GAN_ui.py`

Now you can access the UI at: http://<VM_IP>/
