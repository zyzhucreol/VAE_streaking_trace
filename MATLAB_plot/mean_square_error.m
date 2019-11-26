function mse = mean_square_error( x,y )
%mean_square_error Reduced mean MSE
mse=mean((x(:)-y(:)).^2);


end

