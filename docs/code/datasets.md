**Our complete dataset can be downloaded [here](https://drive.google.com/file/d/1yOgpTvak3-o2EuCrSL3CyWcffeShwHDD/view?usp=sharing)**.

*Note: This dataset consists of 800 Images (sized: 320x320) per font style. The directories are structured in accordance to Input, Testing, and Training Data.*

*In order to create a DataLoader, one may refer [Dataloader.py](https://github.com/AI4Bharat/Fonts-for-Indian-Scripts/blob/main/ProcessFont/Model/Dataloader.py)*.


We have used 800 Images per style due to compute limitations. We believe more training data would result in better results for sure.
To **create more data**, please refer to the following steps:

1. You can either download open devanagari fonts from [Google fonts](fonts.google.com), or already scraped by us [here](https://drive.google.com/file/d/1wJzrUIzr4TRC8YoTmy37kGhBh6R5HMOt/view?usp=sharing).
2. Unzip the contents, save [fonts_gen.py](https://github.com/AI4Bharat/Fonts-for-Indian-Scripts/blob/main/font_gen.py) and [hi_word_list](https://github.com/AI4Bharat/Fonts-for-Indian-Scripts/blob/main/Words%20List/hi_word_list.json) in the same directory.
3. Install all the dependencies listed [here](https://github.com/AI4Bharat/Fonts-for-Indian-Scripts/blob/main/requirements.sh).
4. Change your local paths in [fonts_gen.py](https://github.com/AI4Bharat/Fonts-for-Indian-Scripts/blob/main/font_gen.py).
