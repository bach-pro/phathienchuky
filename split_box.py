import os
import cv2

image_dir = "dataset/images"
label_dir = "dataset/labels"

def rotate_yolo_box(x, y, w, h, mode):
    """
    mode:
        left  : xoay 90° ngược chiều kim đồng hồ
        right : xoay 90° cùng chiều kim đồng hồ
        180   : xoay 180°
    """

    if mode == "left":
        return y, 1 - x, h, w

    elif mode == "right":
        return 1 - y, x, h, w

    elif mode == "180":
        return 1 - x, 1 - y, w, h

    else:
        raise ValueError("Unknown mode")


for img_name in os.listdir(image_dir):

    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(image_dir, img_name)
    label_path = os.path.join(
        label_dir,
        os.path.splitext(img_name)[0] + ".txt"
    )

    img = cv2.imread(img_path)

    if img is None:
        continue

    # đọc label
    with open(label_path, "r") as f:
        labels = [line.strip().split() for line in f if line.strip()]

    rotations = {
        "left": cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
        "right": cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
        "180": cv2.rotate(img, cv2.ROTATE_180),
    }

    for mode, new_img in rotations.items():

        base = os.path.splitext(img_name)[0]

        # lưu ảnh
        new_img_path = os.path.join(
            image_dir,
            f"{base}_{mode}.jpg"
        )
        cv2.imwrite(new_img_path, new_img)

        # lưu label
        new_label_path = os.path.join(
            label_dir,
            f"{base}_{mode}.txt"
        )

        with open(new_label_path, "w") as f:

            for lb in labels:
                cls = lb[0]
                x = float(lb[1])
                y = float(lb[2])
                w = float(lb[3])
                h = float(lb[4])

                nx, ny, nw, nh = rotate_yolo_box(
                    x, y, w, h, mode
                )

                f.write(
                    f"{cls} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}\n"
                )

print("Done!")