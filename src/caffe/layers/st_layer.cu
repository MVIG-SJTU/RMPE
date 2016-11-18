#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"
#include "caffe/layers/st_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
__global__ void set_value_to_constant(const int nthreads, Dtype value, int size, 
	int i, Dtype* dst) {

	CUDA_KERNEL_LOOP(index, nthreads) {
		dst[index * size + i] = value;
	}
}

template <typename Dtype>
__global__ void copy_values(const int nthreads, int size_src, int k, 
	const Dtype* src, int size_dst, int i, Dtype* dst) {

	CUDA_KERNEL_LOOP(index, nthreads) {
		dst[index * size_dst + i] = src[index * size_src + k];
	}
}

template <typename Dtype>
__global__ void SpatialTransformerForwardGPU(const int nthreads, int N, int C,
		int output_H_, int output_W_, int H, int W,
		const Dtype* input_grid_data, const Dtype* U, Dtype* V) {
	
	CUDA_KERNEL_LOOP(index, nthreads) {

		const int t = index % output_W_;
		const int s = (index / output_W_) % output_H_;
		const int j = (index / (output_W_ * output_H_)) % C;
		const int i = index / (output_W_ * output_H_ * C);

		const Dtype* coordinates = input_grid_data + (output_H_ * output_W_ * 2) * i;
		const int row_idx = output_W_ * s + t;

	  	const Dtype px = coordinates[row_idx * 2];
	  	const Dtype py = coordinates[row_idx * 2 + 1];

	  	const int V_offset = index;

	  	V[V_offset] = (Dtype)0.;

	  	const Dtype x = (px + 1) / 2 * H;
	  	const Dtype y = (py + 1) / 2 * W;

	  	int m, n; Dtype w;
	  	const Dtype* pic = U + i * (C * H * W) + j * (H * W);

	  	m = floor(x); n = floor(y); w = 0;
	  	if(m >= 0 && m < H && n >= 0 && n < W) {
	  		w = (1 - (x - m)) * (1 - (y - n));
	  		V[V_offset] += w * pic[m * W + n];
	  	}

	  	m = floor(x) + 1; n = floor(y); w = 0;
	  	if(m >= 0 && m < H && n >= 0 && n < W) {
	  		w = (1 - (m - x)) * (1 - (y - n));
	  		V[V_offset] += w * pic[m * W + n];
	  	}

	  	m = floor(x); n = floor(y) + 1; w = 0;
	  	if(m >= 0 && m < H && n >= 0 && n < W) {
	  		w = (1 - (x - m)) * (1 - (n - y));
	  		V[V_offset] += w * pic[m * W + n];
	  	}

	  	m = floor(x) + 1; n = floor(y) + 1; w = 0;
	  	if(m >= 0 && m < H && n >= 0 && n < W) {
	  		w = (1 - (m - x)) * (1 - (n - y));
	  		V[V_offset] += w * pic[m * W + n];
	  	}
  }
}

template <typename Dtype>
void SpatialTransformerLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	string prefix = "SpatialTransformerLayer::Forward_gpu::\t";

	const Dtype* U = bottom[0]->gpu_data();
	const Dtype* theta = bottom[1]->gpu_data();
	const Dtype* output_grid_data = output_grid.gpu_data();
	
	Dtype* full_theta_data = full_theta.mutable_gpu_data();
	Dtype* full_gamma_data = full_gamma.mutable_cpu_data();
	Dtype* input_grid_data = input_grid.mutable_gpu_data();
	Dtype* V = top[0]->mutable_gpu_data();

	caffe_gpu_set(input_grid.count(), (Dtype)0, input_grid_data);
	caffe_gpu_set(top[0]->count(), (Dtype)0, V);
	
	// compute full_theta
	int k = 0; 
	const int num_threads = N;
	for(int i=0; i<6; ++i) {
		if(is_pre_defined_theta[i]) {
			set_value_to_constant<Dtype><<<CAFFE_GET_BLOCKS(num_threads), CAFFE_CUDA_NUM_THREADS>>>( 
				num_threads, pre_defined_theta[i], 6, i, full_theta_data);
			//std::cout << "Setting value " << pre_defined_theta[i] << " to "<< i << 
			//	"/6 of full_theta_data" << std::endl;
		} else {
			copy_values<Dtype><<<CAFFE_GET_BLOCKS(num_threads), CAFFE_CUDA_NUM_THREADS>>>(num_threads, 
				6 - pre_defined_count, k, theta, 6, i, full_theta_data);
			//std::cout << "Copying " << k << "/" << 6 - pre_defined_count << " of theta to " 
			//	<< i << "/6 of full_theta_data" << std::endl;
			++ k;
		}
	}
	// For detransform, calculate gamma for de-transform
	if(de_transform){
		for(int i=0; i<N; i++){
			double denom_ = full_gamma_data[6*i+0]*full_gamma_data[6*i+4] - full_gamma_data[6*i+1]*full_gamma_data[6*i+3];
			if(denom_ == 0.0){
				DLOG(INFO) << "Singular matrix encountered. Do identity mapping.";
				full_gamma_data[6*i+0] = 1; full_gamma_data[6*i+1] = 0; full_gamma_data[6*i+2] = 0;
				full_gamma_data[6*i+3] = 0; full_gamma_data[6*i+4] = 1; full_gamma_data[6*i+5] = 0;
			}
			else{
				Dtype tmp_a = full_gamma_data[6*i+0];
				Dtype tmp_b = full_gamma_data[6*i+1];
				full_gamma_data[6*i+0] = full_gamma_data[6*i+4]/denom_; full_gamma_data[6*i+1] = full_gamma_data[6*i+3]/denom_; 
				full_gamma_data[6*i+3] = tmp_b/denom_; full_gamma_data[6*i+4] = tmp_a/denom_; 
				
				Dtype tmp_c = full_gamma_data[6*i+2];
				Dtype tmp_d = full_gamma_data[6*i+5];
				full_gamma_data[6*i+2] = -(full_gamma_data[6*i+0]*tmp_c + full_gamma_data[6*i+1]*tmp_d);
				full_gamma_data[6*i+5] = -(full_gamma_data[6*i+3]*tmp_c + full_gamma_data[6*i+4]*tmp_d);
			}
		}
		// compute out input_grid_data
		for(int i = 0; i < N; ++i) {
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, output_H_ * output_W_, 2, 3, (Dtype)1.,
					output_grid_data, full_gamma_data + 6 * i, (Dtype)0.,
					input_grid_data + (output_H_ * output_W_ * 2) * i);
		}
	}
	else{
		// compute out input_grid_data
		for(int i = 0; i < N; ++i) {
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, output_H_ * output_W_, 2, 3, (Dtype)1.,
					output_grid_data, full_theta_data + 6 * i, (Dtype)0.,
					input_grid_data + (output_H_ * output_W_ * 2) * i);
		}
	}

	const int nthreads = N * C * output_H_ * output_W_;

	SpatialTransformerForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
	      CAFFE_CUDA_NUM_THREADS>>>(nthreads, N, C, output_H_, output_W_, H, W, input_grid_data, U, V);
}

template <typename Dtype>
__global__ void SpatialTransformerBackwardGPU_dTheta(const int nthreads, int C,
		int output_H_, int output_W_, int H, int W,
		const Dtype* input_grid_data, const Dtype* dV_array, const Dtype* U_array,  
		Dtype* dTheta_tmp_diff) {
	
	CUDA_KERNEL_LOOP(index, nthreads) {

		const int t = index % output_W_;
		const int s = (index / output_W_) % output_H_;
		const int j = (index / (output_W_ * output_H_)) % C;
		const int i = index / (output_W_ * output_H_ * C);

		const Dtype* coordinates = input_grid_data + (output_H_ * output_W_ * 2) * i;

		const int row_idx = output_W_ * s + t;

		const Dtype px = coordinates[row_idx * 2];
		const Dtype py = coordinates[row_idx * 2 + 1];
		
		Dtype delta_dpx = (Dtype)0.;
		Dtype delta_dpy = (Dtype)0.;

		const Dtype x = (px + 1) / 2 * H;
		const Dtype y = (py + 1) / 2 * W;
		const int dV_offset = index;
		const Dtype dV = dV_array[dV_offset];

		int m, n; 
		const Dtype* U = U_array + i * (C * H * W) + j * (H * W);

		// left-bottom neighbor
		m = floor(x); n = floor(y); 
		if(m >= 0 && m < H && n >= 0 && n < W) {
			delta_dpx -= (1 - (y - n)) * U[m * W + n] * dV * H / 2;
			delta_dpy -= (1 - (x - m)) * U[m * W + n] * dV * W / 2;
		}
		
		// left-top neighbor
		m = floor(x); n = floor(y) + 1; 
		if(m >= 0 && m < H && n >= 0 && n < W) {
			delta_dpx -= (1 - (n - y)) * U[m * W + n] * dV * H / 2;
			delta_dpy += (1 - (x - m)) * U[m * W + n] * dV * W / 2;
		}

		// right-bottom neighbor
		m = floor(x) + 1; n = floor(y); 
		if(m >= 0 && m < H && n >= 0 && n < W) {
			delta_dpx += (1 - (y - n)) * U[m * W + n] * dV * H / 2;
			delta_dpy -= (1 - (m - x)) * U[m * W + n] * dV * W / 2;
		}
		
		// right-top neighbor
		m = floor(x) + 1; n = floor(y) + 1; 
		if(m >= 0 && m < H && n >= 0 && n < W) {
			delta_dpx += (1 - (n - y)) * U[m * W + n] * dV * H / 2;
			delta_dpy += (1 - (m - x)) * U[m * W + n] * dV * W / 2;
		}
		
		int idx = j * (output_H_ * output_W_) + s * output_W_ + t;
		
		dTheta_tmp_diff[(6 * i) * (output_H_ * output_W_ * C) + idx] += delta_dpx * (s * 1.0 / output_H_ * 2 - 1);
		dTheta_tmp_diff[(6 * i + 1) * (output_H_ * output_W_ * C) + idx] += delta_dpx * (t * 1.0 / output_W_ * 2 - 1);
		dTheta_tmp_diff[(6 * i + 2) * (output_H_ * output_W_ * C) + idx] += delta_dpx;
		dTheta_tmp_diff[(6 * i + 3) * (output_H_ * output_W_ * C) + idx] += delta_dpy * (s * 1.0 / output_H_ * 2 - 1);
		dTheta_tmp_diff[(6 * i + 4) * (output_H_ * output_W_ * C) + idx] += delta_dpy * (t * 1.0 / output_W_ * 2 - 1);
		dTheta_tmp_diff[(6 * i + 5) * (output_H_ * output_W_ * C) + idx] += delta_dpy;
	}
}

template <typename Dtype>
__global__ void SpatialTransformerBackwardGPU_dU(const int nthreads, const int C, 
	const int W,  const int H, const int output_H_, const int output_W_, 
	const Dtype* input_grid_data, const Dtype* dV, Dtype* dU) {
	
	CUDA_KERNEL_LOOP(index, nthreads) {

		const int t = index % output_W_;
		const int s = (index / output_W_) % output_H_;
		const int j = (index / (output_W_ * output_H_)) % C;
		const int i = index / (output_W_ * output_H_ * C);

		const Dtype* coordinates = input_grid_data + (output_H_ * output_W_ * 2) * i;
		const int row_idx = output_W_ * s + t;

	  	const Dtype px = coordinates[row_idx * 2];
	  	const Dtype py = coordinates[row_idx * 2 + 1];

	  	const int V_offset = index;

	  	const Dtype x = (px + 1) / 2 * H;
	  	const Dtype y = (py + 1) / 2 * W;

	  	int m, n; Dtype w;
	  	Dtype* pic = dU + i * (C * H * W) + j * (H * W);

	  	m = floor(x); n = floor(y); w = 0;
	  	if(m >= 0 && m < H && n >= 0 && n < W) {
	  		w = (1 - (x - m)) * (1 - (y - n));
			caffe_gpu_atomic_add(w * dV[V_offset], pic + (m * W + n));
	  	}

	  	m = floor(x) + 1; n = floor(y); w = 0;
	  	if(m >= 0 && m < H && n >= 0 && n < W) {
	  		w = (1 - (m - x)) * (1 - (y - n));
			caffe_gpu_atomic_add(w * dV[V_offset], pic + (m * W + n));
	  	}

	  	m = floor(x); n = floor(y) + 1; w = 0;
	  	if(m >= 0 && m < H && n >= 0 && n < W) {
	  		w = (1 - (x - m)) * (1 - (n - y));
			caffe_gpu_atomic_add(w * dV[V_offset], pic + (m * W + n));
	  	}

	  	m = floor(x) + 1; n = floor(y) + 1; w = 0;
	  	if(m >= 0 && m < H && n >= 0 && n < W) {
	  		w = (1 - (m - x)) * (1 - (n - y));
			caffe_gpu_atomic_add(w * dV[V_offset], pic + (m * W + n));
	  	}
	}
}

template <typename Dtype>
void SpatialTransformerLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	string prefix = "SpatialTransformerLayer::Backward_GPU::\t";

	const Dtype* dV = top[0]->gpu_diff();
	const Dtype* input_grid_data = input_grid.gpu_data();
	const Dtype* U = bottom[0]->gpu_data();

	Dtype* dFull_theta = full_theta.mutable_gpu_diff();
	Dtype* dTheta = bottom[1]->mutable_gpu_diff();	
	if(!de_transform){
		Dtype* dTheta_tmp_diff = dTheta_tmp.mutable_gpu_diff();
		
		caffe_gpu_set(dTheta_tmp.count(), (Dtype)0., dTheta_tmp_diff);
		
		const int nthreads = N * C * output_H_ * output_W_;
		
		SpatialTransformerBackwardGPU_dTheta<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
				CAFFE_CUDA_NUM_THREADS>>>(nthreads, C, output_H_, output_W_, H, W, input_grid_data,
						dV, U, dTheta_tmp_diff);
		
		Dtype* all_ones_2_data = all_ones_2.mutable_gpu_data();
		caffe_gpu_set(all_ones_2.count(), (Dtype)1., all_ones_2_data);
			
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, full_theta.count(), 1, output_H_ * output_W_ * C, 
				(Dtype)1., dTheta_tmp_diff, all_ones_2_data, (Dtype)0., dFull_theta);
					
		/*const Dtype* db_dFull_theta = full_theta.cpu_diff();
		for(int i=0; i<full_theta.count(); ++i) {
			std::cout << db_dFull_theta[i] << " ";
		}
		std::cout<<std::endl;*/}
    else{
    	Dtype* dFull_gamma = full_gamma.mutable_gpu_diff();
		Dtype* dGamma = bottom[1]->mutable_gpu_diff();
		Dtype* dGamma_tmp_diff = dGamma_tmp.mutable_gpu_diff();
		Dtype* dTheta_1_2_data = dTheta_1_2.mutable_gpu_data();
		caffe_gpu_set(dGamma_tmp.count(), (Dtype)0., dGamma_tmp_diff);
	
		const int nthreads = N * C * output_H_ * output_W_;
	
		SpatialTransformerBackwardGPU_dTheta<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
				CAFFE_CUDA_NUM_THREADS>>>(nthreads, C, output_H_, output_W_, H, W, input_grid_data,
						dV, U, dGamma_tmp_diff);
	
		Dtype* all_ones_2_data = all_ones_2.mutable_gpu_data();
		caffe_gpu_set(all_ones_2.count(), (Dtype)1., all_ones_2_data);
		
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, full_gamma.count(), 1, output_H_ * output_W_ * C, 
				(Dtype)1., dGamma_tmp_diff, all_ones_2_data, (Dtype)0., dFull_gamma);
				
		/*const Dtype* db_dFull_theta = full_theta.cpu_diff();
		for(int i=0; i<full_theta.count(); ++i) {
			std::cout << db_dFull_theta[i] << " ";
		}
		sstd::cout<<std::endl;*/
            const Dtype* full_theta_data = full_theta.gpu_data();
            const Dtype* full_gamma_data = full_gamma.gpu_data();
            Dtype* dg_dt_data = dg_dt.mutable_gpu_data();
            Dtype* dGamma_1_2_data = dGamma_1_2.mutable_gpu_data();
            for(int i=0; i<N; i++){
                double denom_ = full_theta_data[6*i+0]*full_theta_data[6*i+4] - full_theta_data[6*i+1]*full_theta_data[6*i+3];
                if(denom_ == 0){
                        dFull_theta[6*i+0] = 0; dFull_theta[6*i+1] = 0; dFull_theta[6*i+2] = 0;
                        dFull_theta[6*i+3] = 0; dFull_theta[6*i+4] = 0; dFull_theta[6*i+5] = 0;
                }
                else{
                        //d_theta_3
                        dFull_theta[6*i+2] = -1 * (full_gamma_data[6*i+0]*dGamma[6*i+2] + full_gamma_data[6*i+1]*dGamma[6*i+5]);
                        dFull_theta[6*i+5] = -1 * (full_gamma_data[6*i+3]*dGamma[6*i+2] + full_gamma_data[6*i+4]*dGamma[6*i+5]);

                        //d_theta_1_2
						dg_dt_data[0*4 + 0] = (-1)*full_theta_data[6*i + 4]*full_theta_data[6*i + 4];   dg_dt_data[0*4 + 1] = full_theta_data[6*i + 1]*full_theta_data[6*i + 4];   dg_dt_data[0*4 + 2] = full_theta_data[6*i + 3]*full_theta_data[6*i + 4];  dg_dt_data[0*4 + 3] = (-1)*full_theta_data[6*i + 3]*full_theta_data[6*i + 1];
     					dg_dt_data[1*4 + 0] = full_theta_data[6*i + 4]*full_theta_data[6*i + 3];  dg_dt_data[1*4 + 1] = (-1)*full_theta_data[6*i + 0]*full_theta_data[6*i + 4];   dg_dt_data[1*4 + 2] = (-1)*full_theta_data[6*i + 3]*full_theta_data[6*i + 3];  dg_dt_data[1*4 + 3] = full_theta_data[6*i + 0]*full_theta_data[6*i + 3];
     					dg_dt_data[2*4 + 0] = full_theta_data[6*i + 4]*full_theta_data[6*i + 1];  dg_dt_data[2*4 + 1] = (-1)*full_theta_data[6*i + 1]*full_theta_data[6*i + 1];  dg_dt_data[2*4 + 2] = (-1)*full_theta_data[6*i + 0]*full_theta_data[6*i + 4]; dg_dt_data[2*4 + 3] = full_theta_data[6*i + 0]*full_theta_data[6*i + 1];
     					dg_dt_data[3*4 + 0] = (-1)*full_theta_data[6*i + 3]*full_theta_data[6*i + 1];  dg_dt_data[3*4 + 1] = full_theta_data[6*i + 0]*full_theta_data[6*i + 1];  dg_dt_data[3*4 + 2] = full_theta_data[6*i + 3]*full_theta_data[6*i + 0];  dg_dt_data[3*4 + 3] = (-1)*full_theta_data[6*i + 0]*full_theta_data[6*i + 0];
                		
                		dGamma_1_2_data[0] = dGamma[6*i + 0] - dGamma[6*i + 2]*full_theta_data[6*i + 2];
                		dGamma_1_2_data[1] = dGamma[6*i + 3] - dGamma[6*i + 5]*full_theta_data[6*i + 2];
                		dGamma_1_2_data[2] = dGamma[6*i + 1] - dGamma[6*i + 2]*full_theta_data[6*i + 5];
                		dGamma_1_2_data[3] = dGamma[6*i + 4] - dGamma[6*i + 5]*full_theta_data[6*i + 5];
                		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 4, 4, 1, 
							(Dtype)1., dg_dt_data, dGamma_1_2_data, (Dtype)0., dTheta_1_2_data);

                		dFull_theta[6*i+0] = dTheta_1_2_data[0]; dFull_theta[6*i+1] = dTheta_1_2_data[2];
                		dFull_theta[6*i+3] = dTheta_1_2_data[1]; dFull_theta[6*i+4] = dTheta_1_2_data[3];
                }
            }
    }

	int k = 0;
	const int num_threads = N;
	for(int i=0; i<6; ++i) {
		if(!is_pre_defined_theta[i]) {
			copy_values<Dtype><<<CAFFE_GET_BLOCKS(num_threads), CAFFE_CUDA_NUM_THREADS>>>(num_threads, 
				6, i, dFull_theta, 6 - pre_defined_count, k, dTheta);
			//std::cout << "Copying " << i << "/6 of dFull_theta to " << k << "/" << 
			//	6 - pre_defined_count << " of dTheta" << std::endl;
			++ k;
		}
	}
		
	/*const Dtype* db_dtheta = bottom[1]->cpu_diff();
	for(int i=0; i<bottom[1]->count(); ++i) {
		std::cout << db_dtheta[i] << " ";
	}
	std::cout<<std::endl;*/

	if(to_compute_dU_ or de_transform) {
		Dtype* dU = bottom[0]->mutable_gpu_diff();
		caffe_gpu_set(bottom[0]->count(), (Dtype)0., dU);
		const int nthreads = N * C * output_H_ * output_W_;
		SpatialTransformerBackwardGPU_dU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
			CAFFE_CUDA_NUM_THREADS>>>(nthreads, C, W, H, output_H_, output_W_, input_grid_data, dV, dU);
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(SpatialTransformerLayer);

}	// namespace caffe