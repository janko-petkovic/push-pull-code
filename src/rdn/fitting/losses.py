import torch


class NLLGauss():
    def __init__(self):
        pass

    def __call__(self, F_of_p, data, data_err, p):
        """
        Evaluate negative log likelihood using gaussian noise
        """
        # declare variables
        l_prior = 0.
        l_llh = 0.
        # in case of having prior distributions for the parameters, 
        # these can also be encoded here, e.g. Gaussian distributions
        
        # transform parameter scale
        # p = torch.pow(10, p)
        
        
        # simulate model
        prediction = F_of_p(p)
            
        # constant
        ll_sigma = torch.log(2*torch.pi*data_err**2).sum()
        ll_mse = (((data - prediction)/data_err)**2).sum()

        # evaluate log likelihood
        ll_target = 0.5*(ll_sigma+ll_mse)
        
        # return negative log posterior
        return ll_target

    def __str__(self):
        return 'NLLGauss'



class NLLLaplace():
    def __init__(self):
        pass

    def __call__(self, F_of_p, data, data_err, p):
        """
        Evaluate negative log likelihood using laplace noise
        """
        # declare variables
        l_prior = 0.
        l_llh = 0.
        # in case of having prior distributions for the parameters, 
        # these can also be encoded here, e.g. Gaussian distributions
        
        # transform parameter scale
        # p = torch.pow(10, p)
        
        
        # simulate model
        prediction = F_of_p(p)
            
        # constant
        ll_sigma = torch.log(2*data_err).sum()
        ll_mse = ((data - prediction).abs()/data_err).sum()

        # evaluate log likelihood
        ll_target = 0.5*(ll_sigma+ll_mse)
        
        # return negative log posterior
        return ll_target

    def __str__(self):
        return 'NLLLaplace'



class NLLStudent():
    def __init__(self):
        pass

        
    def __call__(self, F_of_p, data, data_err, p):
        """
        Evaluate negative log likelihood using student noise
        We suppose that on average we have 2 ndof
        """

        ndof = 2

        # declare variables
        l_prior = 0.
        l_llh = 0.
        # in case of having prior distributions for the parameters, 
        # these can also be encoded here, e.g. Gaussian distributions
        
        # transform parameter scale
        # p = torch.pow(10, p)
        
        
        # simulate model
        prediction = F_of_p(p)


        # # constant
        # ll_sigma = torch.log(ndof * data_err**2).sum()
        # ll_mse = ((ndof+1)/2 * torch.log(1 + 1/ndof*((data - prediction)/data_err)**2)).sum()

        # # evaluate log likelihood
        # ll_target = 0.5*(ll_sigma+ll_mse)

        # Log-loss without additive constants
        ll_target = ((ndof+1)/2 * torch.log(1 + 1/ndof*((data - prediction)/data_err)**2)).sum()
        
        # return negative log posterior
        return ll_target

    def __str__(self):
        return 'NLLStudent'




class NLLAdast():
    def __init__(self):
        pass

        
    def __call__(self, F_of_p, data, data_err, ndof,  p):
        """
        Evaluate negative log likelihood using student noise
        We suppose that on average we have 2 ndof
        """

        # declare variables
        l_prior = 0.
        l_llh = 0.
        # in case of having prior distributions for the parameters, 
        # these can also be encoded here, e.g. Gaussian distributions
        
        # simulate model
        prediction = F_of_p(p)

        # log_pred = torch.log(prediction)
        # log_data = torch.log(data)
        # log_err = data_err / data

        # Log-loss without additive constants
        ll_target = ((ndof+1)/2 * torch.log(1 + 1/ndof*((data - prediction)/data_err)**2)).sum()
        # ll_target = ((ndof+1)/2 * torch.log(1 + 1/ndof*((log_data - log_pred)/log_err)**2)).sum()
        
        # return negative log posterior
        return ll_target

    def __str__(self):
        return 'NLLAdast'



class ChiSquared():
    def __init__(self):
        pass

        
    def __call__(self, F_of_p, data, data_err, p):
        """
        Evaluate negative log likelihood using student noise
        We suppose that on average we have 2 ndof
        """

        ndof = 2

        # declare variables
        l_prior = 0.
        l_llh = 0.
        # in case of having prior distributions for the parameters, 
        # these can also be encoded here, e.g. Gaussian distributions
        
        # transform parameter scale
        # p = torch.pow(10, p)
        
        
        # simulate model
        prediction = F_of_p(p)

        # constant
        ll_mse = (((data - prediction)/data_err)**2).sum()
        

        # return negative log posterior
        return ll_mse

    def __str__(self):
        return 'ChiSquared'
