colormap ("default");
 [theta, r] = meshgrid (linspace (0,2*pi,64), linspace (0,1,64));
 [X, Y] = pol2cart (theta, r);
 Z = sin (2*theta) .* (1-r);

figure;
surf(X, Y, Z);
xlabel('\theta_0'); ylabel('\theta_1');

% Contour plot
figure;
 contour (X, Y, abs (Z), 10);
 title ({"contour() plot"; "polar fcn: Z = sin (2*theta) * (1-r)"});