function img_out = crop_img(root_path, target_size, suffix, out_root)
    imgs = dir(strcat(root_path, '*.', suffix));
    num_imgs = length(imgs);
    fprintf("Number of images: %d\n", num_imgs)
    if num_imgs <= 0
        fprintf("ERROR: Wrong input path. %s\n", root_path)
    else
        bar = waitbar(0, 'Croping Images...');
        for i = 1:num_imgs
            img_path = strcat(root_path, imgs(i).name);
            % fprintf("Read: %s\n", img_path);
            img = imread(img_path);

            window = centerCropWindow2d(size(img),target_size);
            img_out = imcrop(img,window);
            % imshow(img_out)

            out_path = strcat(out_root, "crop_", imgs(i).name);
            imwrite(img_out, out_path)
            waitbar(i/num_imgs, bar)
        end
        close(bar);
    end
    fprintf("Croping Images Done!")
end