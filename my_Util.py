public_data_path = "../data"

def imshow_double(img1,img2,label1="",label2=""):
    fig = plt.figure()
    rows = 1
    cols = 2

    ax1 = fig.add_subplot(rows, cols, 1)
    ax1.imshow(img1)
    ax1.set_title(label1)
    ax1.axis("off")

    ax2 = fig.add_subplot(rows, cols,2)
    ax2.imshow(img2)
    ax2.set_title(label2)
    ax2.axis("off")
