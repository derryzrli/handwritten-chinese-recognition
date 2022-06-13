# Traditional Chinese Handwriting Recognition App

### Problem Statement

Handwriting Classification is a well-known domain to most Data and ML Scientists. Inspired by the work done with the MNIST dataset, I want to build an app that aims to recognize traditionally hand-written Chinese characters, known to be one of the most complicated language systems overall. This is particularly helpful to Chinese language learners who may want to use it to look up characters they don't know the pronunciation of, as well as elderlies who may be unfamiliar with the modern Pinyin system used by most keyboard inputs. 


### Content Order

Please review this project in the following order: 
1. [`file_reorg`](./file_reorg.ipynb): Re-organization of the files according to CNN training workflow
2. [`train_small`](./train-small.ipynb): Training on 4 characters - 學 無 止 境 
3. [`train_small`](./train-100.ipynb): Training on the top 100 most commonly used character
4. [`train_small`](./train-full.ipynb): Training on all 4803 characters in the dataset
5. [`gradio`](./gradio.ipynb): Implementation of web app hosted by Gradio


### Dataset Used
* Traditional Chinese Handwriting Dataset: Dataset obtained from AI-FREE's [`git repository`](https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset). (Note: dataset not included in my personal repo due to size restraints)


### The Analysis + Modeling Process

- File Reoganization (rename, create, move) to make suitable for Tensorflow modeling workflow.
- Recreated AI-FREE's [`approach`](https://github.com/AI-FREE-Team/Handwriting-Chinese-Characters-Recognition/blob/master/Traditional_Chinese_CNN_Model_colab.ipynb)
- Reconstructed and altered CNN model's structure such that we obtain one that outperformed the former approach
- Once we improved the model, we apply and train it on more data (i.e. top 100 characters, all 4803 characters)


### Gradio Web App

Check out our [`app on Gradio`](https://58017.gradio.app/)! Reach out to me if it is unaccessible :) 


### Conclusion and Future Work

When training 4 characters, we were able to acheive 97% testing accuracy, much higher than the baseline model of 85% (AI-FREE's approach). 

We then expanded further to train on the top 100 most commonly used characters. With a epochs of 1000, we were able to acheive an impressive 76% overall accuracy on the testing set. We believe more data and long trainig time will further increase the performance.

Finally we applied the same training on all 4803 of our characters. Given the time and hardware constraint, an epoch of 50 took more than 8 hours to train, with only a testing accuracy as low as 20%. More data and training time potentially could help, but it is certainly outside of the scope of this project and my personal hardware limit.

For future work, more varirty of data in handwriting style and stroke thickness, for example, could contribute to the success of a robust model. 


### Acknowledgement

Thanks to the dataset provided by the [`AI-Free team`](https://github.com/AI-FREE-Team), whose work also guided me through the modeling process, I am able to create this app. This personal project adopts some of the code used in file re-organization to ensure model success.




