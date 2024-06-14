import os

import cv2
import numpy as np

def warp_image(img, disparity_map, direction):
    # Image warping
    h, w = img.shape
    warped_img = np.zeros((h, w), dtype=np.int32)
    for y in range(h):
        for x in range(w):
            new_x = x - disparity_map[y, x]      

            if 0 <= new_x < w:
                warped_img[y, x] = img[y, new_x]
            else:
                warped_img[y, x]  = img[y, x]
                print(new_x, x, y)
    
    return warped_img

def aggregate_cost_volume(cost_volume):
    # Hyperparameter
    p1 = 5
    p2 = 150
    r = 4
    
    # Calculate aggregate cost volume
    h, w, max_disparity = cost_volume.shape
    aggregated_costs = np.full((h, w, max_disparity, r), np.inf, dtype=np.float32)
    forward_pass = [(0, -1), (1, -1), (1, 0), (1, 1)]
    backward_pass = [(0, 1), (-1, 1), (-1, 0), (-1, -1)]
    
    # Forward pass
    for idx, (dy, dx) in enumerate(forward_pass):
        for y in range(h - 1, -1, -1):
            for x in range(w):
                for d in range(max_disparity):
                    # D(p, d)
                    aggregated_costs[y, x, d, idx] = cost_volume[y, x, d]

                    ny = y + dy
                    nx = x + dx
                    if not (0 <= ny < h and 0 <= nx < w):
                        continue
                    
                    # min_k(L_r(p - r, k)
                    d_min_cost = aggregated_costs[ny, nx, 0, idx]
                    for dd in range(max_disparity):
                        d_min_cost = min(d_min_cost, aggregated_costs[ny, nx, dd, idx])

                    # min(L_r(p - r, d), L_r(p - r, d - 1) + p1, L_r(p - r, d + 1) + p2, d_min + p2)
                    neighbor_min_cost = aggregated_costs[ny, nx, d, idx]
                    if d - 1 >= 0:
                        neighbor_min_cost = min(neighbor_min_cost, aggregated_costs[ny, nx, d - 1, idx] + p1)
                    if d + 1 < max_disparity:
                        neighbor_min_cost = min(neighbor_min_cost, aggregated_costs[ny, nx, d + 1, idx] + p1)
                    neighbor_min_cost = min(neighbor_min_cost, d_min_cost + p2)

                    if np.isinf(neighbor_min_cost) and np.isinf(d_min_cost):
                        continue
                    aggregated_costs[y, x, d, idx] += neighbor_min_cost - d_min_cost 
    
    # Backward pass
    for idx, (dy, dx) in enumerate(backward_pass):
        for y in range(h):
            for x in range(w - 1, -1, -1):
                for d in range(max_disparity):
                    # D(p, d)
                    aggregated_costs[y, x, d, idx] = cost_volume[y, x, d]

                    ny = y + dy
                    nx = x + dx
                    if not (0 <= ny < h and 0 <= nx < w):
                        continue
                        
                    # min_k(L_r(p - r, k)
                    d_min_cost = aggregated_costs[ny, nx, 0, idx]
                    for dd in range(max_disparity):
                        d_min_cost = min(d_min_cost, aggregated_costs[ny, nx, dd, idx])

                    # min(L_r(p - r, d), L_r(p - r, d - 1) + p1, L_r(p - r, d + 1) + p2, d_min + p2)
                    neighbor_min_cost = aggregated_costs[ny, nx, d, idx]
                    if d - 1 >= 0:
                        neighbor_min_cost = min(neighbor_min_cost, aggregated_costs[ny, nx, d - 1, idx] + p1)
                    if d + 1 < max_disparity:
                        neighbor_min_cost = min(neighbor_min_cost, aggregated_costs[ny, nx, d + 1, idx] + p1)
                    neighbor_min_cost = min(neighbor_min_cost, d_min_cost + p2)

                    if np.isinf(neighbor_min_cost) and np.isinf(d_min_cost):
                        continue
                    aggregated_costs[y, x, d, idx] += neighbor_min_cost - d_min_cost 
    
    aggregated_volume = np.sum(aggregated_costs, axis=3)
    return aggregated_volume

def SAD(img1, img2):
    # Calculate SAD (Sum of Absolute Differences)
    img1 = np.array(img1, dtype=np.int32)
    img2 = np.array(img2, dtype=np.int32)
    abs_diff = np.abs(img1 - img2)
    
    return abs_diff

def similarity(left_img, right_img, max_disparity, direction):
    # Measure pixel-wise similarity using distance function
    h, w = left_img.shape
    cost_volume = np.full((h, w, max_disparity), np.inf, dtype=np.float32)
    cost_volume[:, :, 0] = SAD(left_img, right_img)
    for disparity in range(1, max_disparity):
        cost_volume[:, disparity:, disparity] = SAD(left_img[:, disparity:], right_img[:, :-1*disparity])
    disparity_map = cost_volume.argmin(axis=2)
    
    return cost_volume, disparity_map

def normalize(x):
    # normalize image
    return ((x - x.min()) / (x.max() - x.min()) * 255).astype(np.uint8)

def semi_global_matching(left_img, right_img, max_disparity, direction):
    # similarity
    cost_volume, disparity_map = similarity(left_img, right_img, max_disparity, direction)
    
    # dp cost volume
    aggregated_cost_volume = aggregate_cost_volume(cost_volume)
    aggregated_disparity_map = aggregated_cost_volume.argmin(axis=2)
    
    return cost_volume, disparity_map, aggregated_disparity_map
    
if __name__ == '__main__':
    # dir
    img_dir = 'input'
    output_dir = 'output'
    target_dir = 'target'
    debug = False
    
    # hyperparameter
    max_disparity = 24
    
    # load images
    boundary_range = 100
    img1 = cv2.imread(os.path.join('./', img_dir, '01_noise25.png'), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(os.path.join('./', img_dir, '02_noise25.png'), cv2.IMREAD_GRAYSCALE)
    img3 = cv2.imread(os.path.join('./', img_dir, '03_noise25.png'), cv2.IMREAD_GRAYSCALE)
    img4 = cv2.imread(os.path.join('./', img_dir, '04_noise25.png'), cv2.IMREAD_GRAYSCALE)
    img5 = cv2.imread(os.path.join('./', img_dir, '05_noise25.png'), cv2.IMREAD_GRAYSCALE)
    img6 = cv2.imread(os.path.join('./', img_dir, '06_noise25.png'), cv2.IMREAD_GRAYSCALE)
    img7 = cv2.imread(os.path.join('./', img_dir, '07_noise25.png'), cv2.IMREAD_GRAYSCALE)
    ground_truth = cv2.imread(os.path.join('./', target_dir, 'gt.png'), cv2.IMREAD_GRAYSCALE)
    
    # for debug
    if debug:    
        img1 = img1[boundary_range:-boundary_range, boundary_range:-boundary_range]
        img2 = img2[boundary_range:-boundary_range, boundary_range:-boundary_range]
        img3 = img3[boundary_range:-boundary_range, boundary_range:-boundary_range]
        img4 = img4[boundary_range:-boundary_range, boundary_range:-boundary_range]
        img5 = img5[boundary_range:-boundary_range, boundary_range:-boundary_range]
        img6 = img6[boundary_range:-boundary_range, boundary_range:-boundary_range]
        img7 = img7[boundary_range:-boundary_range, boundary_range:-boundary_range]
        ground_truth = ground_truth[boundary_range:-boundary_range, boundary_range:-boundary_range]

    warped_imgs_dm = [img4]
    warped_imgs_adm = [img4]
    cv2.imwrite(os.path.join('./', output_dir, 'Warped_Image_DM', '04_warped_image.png'), normalize(img4))
    cv2.imwrite(os.path.join('./', output_dir, 'Warped_Image_ADM', '04_warped_image.png'), normalize(img4))
    
    # For left images
    left_imgs = [img1, img2, img3]
    for i, img in enumerate(left_imgs):
        # SGM
        cost_volume, disparity_map, aggregated_disparity_map = semi_global_matching(img, img4, max_disparity, 0)
        np.savetxt(os.path.join('./', output_dir, 'Cost', '0' + str(i + 1) + '_cost_volume.txt'), cost_volume.reshape(-1, max_disparity), fmt='%f', delimiter=' ')
        cv2.imwrite(os.path.join('./', output_dir, 'Intermediate_Disparity', '0' + str(i + 1) + '_disparity_map.png'), normalize(disparity_map))
        cv2.imwrite(os.path.join('./', output_dir, 'Final_Disparity', '0' + str(i + 1) + '_disparity_map.png'), normalize(aggregated_disparity_map))
        
        # Warping
        warped_img_dm = warp_image(img4, disparity_map, 0)
        warped_img_adm = warp_image(img4, aggregated_disparity_map, 0)
        warped_imgs_dm.append(warped_img_dm)
        warped_imgs_adm.append(warped_img_adm)
        cv2.imwrite(os.path.join('./', output_dir, 'Warped_Image_DM', '0' + str(i + 1) + '_warped_image.png'), normalize(warped_img_dm))
        cv2.imwrite(os.path.join('./', output_dir, 'Warped_Image_ADM', '0' + str(i + 1) + '_warped_image.png'), normalize(warped_img_adm))
        print(i + 1, 'Saved')
    
    # For right images
    right_imgs = [img5, img6, img7]
    for i, img in enumerate(right_imgs):
        # SGM
        cost_volume, disparity_map, aggregated_disparity_map = semi_global_matching(img4, img, max_disparity, 1)
        np.savetxt(os.path.join('./', output_dir, 'Cost', '0' + str(i + 5) + '_cost_volume.txt'), cost_volume.reshape(-1, max_disparity), fmt='%f', delimiter=' ')
        cv2.imwrite(os.path.join('./', output_dir, 'Intermediate_Disparity', '0' + str(i + 5) + '_disparity_map.png'), normalize(disparity_map))
        cv2.imwrite(os.path.join('./', output_dir, 'Final_Disparity', '0' + str(i + 5) + '_disparity_map.png'), normalize(aggregated_disparity_map))
        
        # Warping
        warped_img_dm = warp_image(img4, disparity_map, 1)
        warped_img_adm = warp_image(img4, aggregated_disparity_map, 1)
        warped_imgs_dm.append(warped_img_dm)
        warped_imgs_adm.append(warped_img_adm)
        cv2.imwrite(os.path.join('./', output_dir, 'Warped_Image_DM', '0' + str(i + 5) + '_warped_image.png'), normalize(warped_img_dm))
        cv2.imwrite(os.path.join('./', output_dir, 'Warped_Image_ADM', '0' + str(i + 5) + '_warped_image.png'), normalize(warped_img_adm))
        print(i + 5, 'Saved')

    # Aggregate warped images
    aggregated_img_dm = np.mean(warped_imgs_dm, axis=0)
    cv2.imwrite(os.path.join('./', output_dir, 'aggregated_dm.png'), normalize(aggregated_img_dm))
    
    aggregated_img_adm = np.mean(warped_imgs_adm, axis=0)
    cv2.imwrite(os.path.join('./', output_dir, 'aggregated_adm.png'), normalize(aggregated_img_adm))

    # Compute MSE and PSNR
    h, w = ground_truth.shape
    
    noisy = [img1, img2, img3, img4, img5, img6, img7]
    noisy = np.mean(noisy, axis=0)

    ground_truth = normalize(ground_truth)
    aggregated_img_dm = normalize(aggregated_img_dm)
    aggregated_img_adm = normalize(aggregated_img_adm)
    noisy = normalize(noisy)
    
    mse = np.mean((ground_truth - aggregated_img_adm) ** 2)
    psnr = 10 * np.log10((255**2) / mse)
    print("mse with adm: {mse}".format(mse=mse))
    print("psnr with adm: {psnr}".format(psnr=psnr))
    
