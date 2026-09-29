<ul>
    <li>Generative Post Training Quantization(GPTQ) INT4 based on the original GPTQ paper </li>
    <li>Actively Updating</li>
    <li>is not clean but works</li>
    <li>you can run all the cells on kaggle</li?>
    <li>Needs tidying up and comapring against the  quantized Qwen model from hugging face</li>
    <li>I will be publishing this to HuggingFace eventually</li>
</ul>

<h2>The Literature Review for GPTQ</h2>

<ul>
    <li>gptq performs layer by layer quantization</li>
</ul>

<p>OBQ treats each row separately. For each row, one weight is quantized that provides least error when it is quantized. This is based on the diagonal of the inverse of the current hessian. A large value in the diagonal means that it's error can be absorbed by other weights</p>

<p>
    For each quantized weight in position p, the inverse of hessian should be updated. But as they are not in same col, for m rows in, m new inverse hessian matrices will be created. So, each weight creates a new hessian inverse matrix of size (d *d), where d = number of unquanitzed weights. This needs to be done for all the weights in the row. The space complexity  = O(m * d**2)
    The time complexity = O(m * d**3)

    The memory doesn't multiply by 'd' because the old matrices are discarded when a new one is formed.

    For a Qwen3 4B model's q_proj layer, whose shape is (4096, 2560), the peak memory needed would be 4096 * (2560**2) * 2(fp16) ~= 50GB. This is for a single layer. So quantization using OBQ requires massive memory. This is a MEMORY CAPACITY BOUND most of the time.
</p>

<p>
    
</p>