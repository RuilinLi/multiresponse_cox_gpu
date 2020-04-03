#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include "gpu_func.h"
#include <sys/time.h>
#include "proxgpu_types.h"


// Compute the gradient at B
numeric get_value_only(cox_cache &dev_cache,
                  cox_data &dev_data,
                  cox_param &dev_param,
                  numeric *B,
                  int *ncase,
                  int *ncase_cumu,
                  int K,
                  int p,
                  cublasHandle_t handle,
                  cudaStream_t *streams,
                  numeric *cox_val_host)
{
    numeric result = 0.0;
    for (int k = 0; k < K; ++k)
    {
        int n = ncase[k];
        int offset = ncase_cumu[k];
        compute_product(dev_data.X+offset*p, B+p*k, dev_cache.eta+offset, n, p, streams[k], handle,CUBLAS_OP_N);
        apply_exp(dev_cache.eta+offset, dev_cache.exp_eta+offset, n, streams[k]);
        // Save rev_cumsum result to dev_cache.outer_accumu to avoid more cache variables
        rev_cumsum(dev_cache.exp_eta+offset, dev_cache.outer_accumu+offset ,n, streams[k]);
        adjust_ties(dev_cache.outer_accumu+offset, dev_data.rankmin+offset, dev_cache.exp_accumu+offset, n, streams[k]);

        get_coxvalue(dev_cache.exp_accumu+offset, dev_cache.eta+offset, dev_data.censor+offset, dev_cache.cox_val+k, n, streams[k]);
        cudaMemcpyAsync(cox_val_host+k, dev_cache.cox_val+k, sizeof(numeric)*1, cudaMemcpyDeviceToHost, streams[k]);
    }

    cudaDeviceSynchronize();
    for(int k = 0; k < K; ++k)
    {
        result += cox_val_host[k];
    }
    return result;
}



// Compute the gradient at B and save the result to dev_grad
numeric get_gradient(cox_cache &dev_cache,
                  cox_data &dev_data,
                  cox_param &dev_param,
                  numeric *dev_grad,
                  numeric *B,
                  int *ncase,
                  int *ncase_cumu,
                  int K,
                  int p,
                  cublasHandle_t handle,
                  cudaStream_t *streams,
                  bool get_val = false,
                  numeric *cox_val_host=0)
{
    for(int k = 0; k <K; ++k)
    {
        int n = ncase[k];
        int offset = ncase_cumu[k];
        compute_product(dev_data.X+offset*p, B+p*k, dev_cache.eta+offset, n, p, streams[k], handle,CUBLAS_OP_N);
        apply_exp(dev_cache.eta+offset, dev_cache.exp_eta+offset, n, streams[k]);
        // Save rev_cumsum result to dev_cache.outer_accumu to avoid more cache variables
        rev_cumsum(dev_cache.exp_eta+offset, dev_cache.outer_accumu+offset ,n, streams[k]);
        adjust_ties(dev_cache.outer_accumu+offset, dev_data.rankmin+offset, dev_cache.exp_accumu+offset, n, streams[k]);
        // Above is  _update_exp()
        // Below is _update_outer()
        // Save the result of division to residual to avoid more cache variables
        cwise_div(dev_data.censor+offset, dev_cache.exp_accumu+offset, dev_cache.residual+offset,  n, streams[k]);
        cumsum(dev_cache.residual+offset, n, streams[k]);
        adjust_ties(dev_cache.residual+offset, dev_data.rankmax+offset, dev_cache.outer_accumu+offset, n, streams[k]);
        mult_add(dev_cache.residual+offset,
                 dev_cache.exp_eta+offset, 
                 dev_cache.outer_accumu+offset, 
                 dev_data.censor+offset, 
                 n, streams[k]);
        // residual ready
        compute_product(dev_data.X+offset*p, 
                        dev_cache.residual+offset, 
                        dev_grad + k*p, n, p, streams[k], handle, CUBLAS_OP_T);
        // Gradient ready
        // get_val will modify eta, but it's fine
        if(get_val)
        {
            get_coxvalue(dev_cache.exp_accumu+offset, dev_cache.eta+offset, dev_data.censor+offset, dev_cache.cox_val+k, n, streams[k]);
            cudaMemcpyAsync(cox_val_host+k, dev_cache.cox_val+k, sizeof(numeric)*1, cudaMemcpyDeviceToHost, streams[k]);
        }
    }
    numeric result = 0.0;
    if (get_val)
    {
        cudaDeviceSynchronize();
        for(int k = 0; k < K; ++k)
        {
            result += cox_val_host[k];
        }

    }
    return result;
}


// [[Rcpp::export]]
Rcpp::List solve_path(const Rcpp::List & X_list,
                        const Rcpp::List & censoring_list,
                        MatrixXd B,
                        const Rcpp::List & rankmin_list,
                        const Rcpp::List & rankmax_list,
                        double step_size,
                        VectorXd lambda_1_all,
                        VectorXd lambda_2_all,
                        Eigen::RowVectorXd penalty_factor, // Penalty factor for each group of variables
                        int niter, // Maximum number of iterations
                        double linesearch_beta,
                        double eps, // convergence criteria
                        double tol = 1e-10// line search tolerance
                        )
{
    // B is a long and skinny matrix now! rows are features and cols are responses
    const int p = B.rows();
    const int K = B.cols();
    // Create CUDA streams and handle
    cudaStream_t *streams = (cudaStream_t *) malloc(K * sizeof(cudaStream_t));
    for (int k = 0; k<K; ++k)
    {
        cudaStreamCreate(&streams[k]);
    }
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaStream_t copy_stream;
    cudaStream_t nest_stream;
    cudaStreamCreate(&copy_stream);
    cudaStreamCreate(&nest_stream);

    cox_data dev_data;
    cox_cache dev_cache;
    cox_param dev_param;
    numeric *cox_val_host=(numeric *)malloc(sizeof(numeric)*K);
    MatrixXd host_B(p,K);

    int *ncase_cumu = (int *)malloc(sizeof(int)*(K+1));
    ncase_cumu[0] = 0;
    int *ncase = (int*)malloc(sizeof(int)*K);
    std::vector<MapMatd> X_all;
    std::vector<MapVecd> censor; // Modify this in R
    std::vector<MapVeci> rankmin;
    std::vector<MapVeci> rankmax;
    for (int k = 0; k<K; ++k)
    {
        X_all.emplace_back(Rcpp::as<MapMatd>(X_list[k]));
        censor.emplace_back(Rcpp::as<MapVecd>(censoring_list[k]));
        rankmin.emplace_back(Rcpp::as<MapVeci>(rankmin_list[k]));
        rankmax.emplace_back(Rcpp::as<MapVeci>(rankmax_list[k]));
        ncase[k] = X_all[k].rows();
        ncase_cumu[k+1] = ncase_cumu[k] + ncase[k];
    }

    allocate_device_memory(dev_data, dev_cache, dev_param, ncase_cumu[K], K, p);

    // initialize parameters on the device
    cudaMemcpy(dev_param.B, &B(0,0), sizeof(numeric)*K*p, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_param.v, &B(0,0), sizeof(numeric)*K*p, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_param.penalty_factor, &penalty_factor(0), sizeof(numeric)*p, cudaMemcpyHostToDevice);

    // Copy the data
    for(int k = 0; k <K; ++k)
    {
        int n = ncase[k];
        cudaMemcpyAsync(dev_data.X+ncase_cumu[k]*p, &(X_all[k](0,0)), sizeof(numeric) * p *n, cudaMemcpyHostToDevice, streams[k]);
        cudaMemcpyAsync(dev_data.censor+ncase_cumu[k], &(censor[k][0]), sizeof(numeric) *n, cudaMemcpyHostToDevice, streams[k]);
        cudaMemcpyAsync(dev_data.rankmin+ncase_cumu[k], &(rankmin[k][0]), sizeof(int)*n, cudaMemcpyHostToDevice, streams[k]);
        cudaMemcpyAsync(dev_data.rankmax+ncase_cumu[k], &(rankmax[k][0]), sizeof(int)*n, cudaMemcpyHostToDevice, streams[k]); 
    }

    numeric cox_val;
    numeric cox_val_next;
    numeric rhs_ls; // right-hand side of line search condition
    numeric diff; // Max norm of the difference of two consecutive iterates
    numeric lambda_1;
    numeric lambda_2;
    const int num_lambda = lambda_1_all.size();
    numeric step_size_intial = step_size;
    Rcpp::List result(num_lambda);
    bool stop; // Stop line searching
    numeric weight_old, weight_new;
    // Initialization done, starting solving the path
    struct timeval start, end;
    for (int lam_ind = 0; lam_ind < num_lambda; ++lam_ind){
        gettimeofday(&start, NULL);

        lambda_1 = lambda_1_all[lam_ind];
        lambda_2 = lambda_2_all[lam_ind];
        weight_old = 1.0;
        step_size = step_size_intial;
        // Inner iteration
        for (int i = 0; i < niter; ++i)
        {

            // Set prev_B = B
            cublas_copy(dev_param, K*p, copy_stream, handle);
            // Wait for Nesterov weight update
            cudaStreamSynchronize(nest_stream);
            // Update the gradient at v, compute cox_val at v
            cox_val = get_gradient(dev_cache,
                                    dev_data,
                                    dev_param,
                                    dev_param.grad,
                                    dev_param.v,
                                    ncase,
                                    ncase_cumu,
                                    K,
                                    p,
                                    handle,
                                    streams,
                                    true,
                                    cox_val_host);
            
            // Enter line search
            while(true)
            {
                // Update  B
                update_parameters(dev_param,
                                    K,
                                    p,
                                    step_size,
                                    lambda_1,
                                    lambda_2);

                // Get cox_val at updated B
                cox_val_next = get_value_only(dev_cache,
                                                dev_data,
                                                dev_param,
                                                dev_param.B,
                                                ncase,
                                                ncase_cumu,
                                                K,
                                                p,
                                                handle,
                                                streams,
                                                cox_val_host);

                stop = false;
                // This block are the line search conditions
                if(abs((cox_val_next - cox_val)/fmax(1.0, abs(cox_val_next))) > tol){
                    rhs_ls = cox_val + ls_stop_v1(dev_param, step_size,K,p);
                    stop = (cox_val_next <= rhs_ls);
                } else 
                {
                    get_gradient(dev_cache,
                                dev_data,
                                dev_param,
                                dev_param.grad_ls,
                                dev_param.B,
                                ncase,
                                ncase_cumu,
                                K,
                                p,
                                handle,
                                streams);
                    rhs_ls = ls_stop_v2(dev_param, step_size,K,p);
                    stop = (rhs_ls >= 0);

                }

                if (stop)
                {
                    break;
                }
                step_size /= linesearch_beta;
            }

            diff = max_diff(dev_param, K, p);
            if (diff < eps)
            {
                std::cout << "convergence based on parameter change reached in " << i <<" iterations\n";
                std::cout << "current step size is " << step_size << std::endl;
                gettimeofday(&end, NULL);
                double delta  = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
                std::cout <<  "elapsed time is " << delta << " seconds" << std::endl;
                Rcpp::checkUserInterrupt();
                break;
            }

             // Nesterov weight
            weight_new = 0.5*(1+sqrt(1+4*weight_old*weight_old));
            nesterov_update(dev_param,K,p, weight_old, weight_new, nest_stream, handle);
            weight_old = weight_new;

            if (i != 0 && i % 100 == 0)
            {
                std::cout << "reached " << i << " iterations\n";
                gettimeofday(&end, NULL);
                double delta  = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
                std::cout <<  "elapsed time is " << delta  << " seconds" << std::endl;
                Rcpp::checkUserInterrupt();
            }

        }
        cudaMemcpy(&host_B(0,0), dev_param.B, sizeof(numeric)*K*p, cudaMemcpyDeviceToHost);
        result[lam_ind] = host_B;
        std::cout << "Solution for the " <<  lam_ind+1 << "th lambda pair is obtained\n";
    }




    free_device_memory(dev_data, dev_cache, dev_param);
    cublasDestroy(handle);
    for (int si = 0; si< K;++si){
        cudaStreamDestroy(streams[si]);
    }
    cudaStreamDestroy(copy_stream);
    cudaStreamDestroy(nest_stream);
    free(streams);
    free(ncase_cumu);
    free(ncase);
    return result;
}