from tensorflow.image import psnr, ssim

def ssim(y_pred, y_true):
  return ssim(y_true, y_pred , max_val=2.0)

def psnr(y_pred, y_true):
  return psnr(y_true, y_pred , max_val=1.0)