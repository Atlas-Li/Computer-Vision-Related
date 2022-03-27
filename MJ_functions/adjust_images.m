function img_out = adjust_images(root_path, suffix, out_root)
    imgs = dir(strcat(root_path, '*.', suffix));
    num_imgs = length(imgs);
    fprintf("Number of images: %d\n", num_imgs)
    if num_imgs <= 0
        fprintf("ERROR: Wrong input path. %s\n", root_path)
    else
        bar = waitbar(0, 'Adjusting images...');
        for i = 1:num_imgs
            img_path = strcat(root_path, imgs(i).name);
            % fprintf("Read: %s\n", img_path);
            img = imread(img_path);
            
            img_out = imadjust(img);
            % imshow(img_out)

            out_path = strcat(out_root, "vis_", imgs(i).name);
            imwrite(img_out, out_path)
            waitbar(i/num_imgs, bar)
        end
        close(bar);
    end
    fprintf("Adjust images Done!")
end