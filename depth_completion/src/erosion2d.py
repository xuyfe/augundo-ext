import torch


class Erosion2d(torch.nn.Module):
    '''
    Class to perform binary erosion

    Arg(s):
        device : torch.device
            device for erosion
    '''

    def __init__(self, device=torch.device('cuda')):
        super(Erosion2d, self).__init__()
        self.device = device

        # Kernel sizes ordered from largest to smallest, only support 3x3 kernel
        self.kernels = [
            # For padding consistancy please use kernels of the same size; odd number (e.g. 3x3)
            torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], device=self.device),  # 3x3
            torch.tensor([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]], device=self.device),  # 3x2
            torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], device=self.device),  # 2x3
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]], device=self.device),  # 2x2
            torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=self.device),  # 3x1
            torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], device=self.device),  # 1x3
            torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], device=self.device),  # 2x1
            torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]], device=self.device),  # 1x2
        ]

    def erode(self, x, kernel):
        '''
        Applies the erosion kernel to the input

        Arg(s):
            x : torch.Tensor[float32]
                N x 1 x H x W image
            kernel : torch.Tensor[float32]
                k x k convolutional kernel
        Returns:
            N x 1 x H x W eroded image
        '''
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        # Erode the input image with the current kernel
        # Brute force approach (add more kernels if needed above)
        eroded = torch.nn.functional.conv2d(x, kernel, padding=(1, 1))

        return eroded

    def mark(self, eroded, original_kernel):
        '''
        Mark the locations that have been eroded

        Arg(s):
            eroded : torch.Tensor[float32]
                N x 1 x H x W image
            original_kernel : torch.Tensor[float32]
                k x k convolutional kernel used to erode the image
        Returns:
            N x 1 x H x W binary mask
        '''

        dilation_kernel = (original_kernel > 0).float()
        dilation_kernel = dilation_kernel.flip([0, 1])

        dilation_kernel = dilation_kernel.unsqueeze(0).unsqueeze(0)
        dilation_result = torch.nn.functional.conv2d(eroded, dilation_kernel, padding=(1, 1))
        dilation_result = dilation_result > 0

        return dilation_result.float()

    def forward(self, x, return_single_point=True):
        '''
        Erodes the input

        Arg(s):
            x : torch.Tensor[float32]
                N x 1 x H x W image
            return_single_point : bool
                if set, then return single points
        Returns:
            N x 1 x H x W eroded image
        '''

        x = x.to(self.device)
        output = torch.zeros_like(x)

        for kernel in self.kernels:
            eroded = self.erode(x, kernel)
            kernel_sum = kernel.sum().item()
            mask = (eroded >= kernel_sum).float()

            # Sum the erosion outputs
            output = torch.max(output, mask)

            # Mark visited regions
            visited = self.mark(mask, kernel)

            # Now we can exclude all kernel-size components
            # we have found and continue with a new x image
            # and smaller kernels (so there is no overlap)
            x = x - visited

        if return_single_point:
            # Get sum of points in each region
            single_point_map = torch.nn.functional.conv2d(
                output,
                self.kernels[0].unsqueeze(0).unsqueeze(0),
                padding='same')

            # Keep single points
            output = torch.where(
                torch.logical_and(single_point_map > 0, single_point_map < 2),
                torch.ones_like(single_point_map),
                torch.zeros_like(single_point_map))

        return output
