import torch
import pickle
import numpy
from utils import read_pickle

def matrix_factorization(input_file="./dataset/UM_dictionary.pkl",
                         output_file="./dataset/matrix_factorized.pkl"):


    data = read_pickle(input_file)

    matrix=torch.from_numpy(data["matrix"])

    user_num=matrix.shape[0]
    movie_num=matrix.shape[1]
    genre_num=10

    #torch.rand randomly samples from [0,1]
    U=torch.nn.Parameter(torch.rand(user_num,genre_num ))
    V=torch.nn.Parameter(torch.rand(genre_num, movie_num))
    print("Matrix U and V shape")
    print(U.shape)
    print(V.shape)
    # u_optim=torch.optim.SGD(U.)
    # v_optim=torch.optim.SGD(V)
    total_iterations=100000
    lr=1e-1
    total_movies_rated=(matrix>0).sum()
    for iteration in range(total_iterations):
        error=(matrix-torch.mm(U,V))[matrix>0]
        loss=error.pow(2).sum()/total_movies_rated
        loss.backward()

        U.data=U.data-lr*U.grad.data
        V.data=V.data-lr*V.grad.data

        U.grad.data=0*U.grad.data
        V.grad.data=0*V.grad.data
        #lr=lr*0.995
        if iteration%10==0:
           print(f"iter: {iteration}  cost:  {total_movies_rated*loss.item()}")

        MF={}
        MF["U"]=U.detach().numpy()
        MF["V"] = V.detach().numpy()


        f = open(output_file, "wb")

        # write the python object (dict) to pickle file
        pickle.dump(MF, f)

        # close file
        f.close()



