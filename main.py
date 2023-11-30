
from pixtopix.main import main


#def main(args):
#    all_images = []
#    data_path = load_online_dataset(dataset='facades')
#    input_image, real_image = split_image(
#        load_image(str(data_path / 'train/100.jpg')))
#    all_images += random_jitter(input_image, real_image)
#    input_image, real_image = split_image(
#        load_image(str(data_path / 'train/101.jpg')))
#    all_images += random_jitter(input_image, real_image)
#    show_image(*all_images)


if __name__ == '__main__':
    main()
    #test_generate_images(sys.argv[1])
