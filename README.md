# Classic HDR Reconstruction

This project implements the classic HDR image reconstruction algorithm from multiple exposure LDR images, based on the method proposed by Debevec et al. 

This project is an assignment of course "计算摄像学：成像模型理论与深度学习实践" from Peking University

The output is a `.hdr` high dynamic range image, and OpenCV is used for tone mapping and visualization, which outputs a `.jpg`  image.

## Project Structure

```
HDR_Reconstruction/
├── data/                   # Contains exposure image sets and exposure time files
│   └── 06/
│       ├── image1.jpg      # LDR images
|       ...
│       └── exposures.txt   # 
├── src/
│   └── utils.py            # Helper functions
|   └── main.py             # Main script: HDR merging and tone mapping
├── result/                 # Output HDR files (.hdr) and tonemapped results (.jpg)
├── report/                 # lab report
└── README.md               # This project documentation
```

## Dependencies

- Python 3.7+
- OpenCV (`opencv-python`)
- NumPy
- PIL (`pillow`)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script from the project root:

```bash
python main.py --path 06 --save_name hdr_result.jpg --ldr_num 9
```

The copyright of the original data (01-05) from the course belongs to PKU-CameraLab, so they are not released

Two sets of self-captured LDR images (06&07) is available, but they are of low quality

Arguments:
- `--path`: subfolder name under `data/`
- `--save_name`: output filename for the tonemapped LDR image
- `--ldr_num`: the number of used LDR image
- The HDR `.hdr` file will be saved as `result/<path>/output.hdr`

## Features

- Estimate camera response curve using least squares optimization
- Merge multiple LDRs into linear HDR image
- Support Reinhard and Mantiuk tone mapping
- Optional white balance, LAB correction, brightness enhancement
- Export `.hdr` for objective evaluation using HDR-VDP

## Lab Report

See `report/` for LaTeX source. Compile to PDF for submission.

## References

- Debevec, P. E., & Malik, J. (1997). Recovering high dynamic range radiance maps from photographs. *SIGGRAPH*
- Computational Photography course assignment source: https://github.com/PKU-CameraLab/TextBook/releases/download/assignment-8/CP_assignment_8.zip
- HDR-VDP2: https://github.com/hdrlab/hdrvdp
- OpenCV HDR Docs: https://docs.opencv.org