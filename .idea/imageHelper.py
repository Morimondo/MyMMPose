import cv2
import numpy as np
from PIL import Image


class ImageHelper:

    # CV画像を取得
    def getCVImage(self, path):
        return cv2.imread(path)
    # Pillow画像を取得
    def getPilImage(self, path):
        return Image.open(path)

    def save(self, path, image):
        '''画像を保存する'''
        cv2.imwrite(path, image)

    # 画像サイズ変更
    def resizeImage(self, width, height, path=None, image=None):
        # リサイズ前の画像を読み込み
        if path is not None:
            resultImage = Image.open(path)
        elif image is not None:
            resultImage = image

        # 横幅が大きければリサイズする
        if (resultImage.width > int(width)):
            resultImage = resultImage.resize((int(width), resultImage.height))
        # 縦幅が大きければリサイズする
        if (resultImage.height > int(height)):
            resultImage = resultImage.resize((resultImage.width, int(height)))

        return resultImage

    def getReflectorPos(self, image, threshold=225):
        '''反射板の位置を取得'''
        #グレースケールにする
        im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #閾値より白ければ反射板とする
        ref_pos = np.where(im_gray > threshold)
        result = []
        #整理する
        for x, y in zip(ref_pos[0], ref_pos[1]):
            result.append((x, y))

        return result

    def trimming(self, image, left=0, right=0, top=0, bottom=0):
        '''画像のトリミング'''
        w, h = image.size
        img_resize = image.crop((left, top, w - right, h - bottom))
        return img_resize

    def getCaptureImage(self, deviceId=0):
        '''カメラから画像を取得する'''
        cap = cv2.VideoCapture(deviceId)
        #画像を取得
        ret, frame = cap.read()
        #取得できなかったとき
        if not ret:
            print("Caputure Device Not found")
            return null
        return frame


# PillowをCVに変換
    def pilTocv(self, image):
        ''' PIL型 -> OpenCV型 '''
        #最初からOpenCV型なら変更しない
        if type(image) is np.ndarray:
            return image
        new_image = np.array(image, dtype=np.uint8)
        if new_image.ndim == 2:  # モノクロ
            pass
        elif new_image.shape[2] == 3:  # カラー
            new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
        elif new_image.shape[2] == 4:  # 透過
            new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
        return new_image

    # CVをPillowに変換
    def cv2pil(self, image):
        ''' OpenCV型 -> PIL型 '''
        #最初からPIL型なら変更しない
        if type(image) is PIL.Image.Image:
            return image
        new_image = image.copy()
        if new_image.ndim == 2:  # モノクロ
            pass
        elif new_image.shape[2] == 3:  # カラー
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
        elif new_image.shape[2] == 4:  # 透過
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
        new_image = Image.fromarray(new_image)
        return new_image