function enhanced_bundle = turbozoom_image(image_bundle)
%turbozoom_image - Description
%
% Syntax: removed_background = turbozoom_image(images)
%
% Based on "Fast and Simple Super-resolution with Single Images", by Paul Eilers and Cyril Ruckebusch

% Fix parameters (TODO: pass as argument)
fsup = 3;
kappa = 0.000001;
lambda = 30;
delta = 1e-2;

[nx, ny, n] = size(image_bundle);
enhanced_bundle = zeros([fsup*nx fsup*ny n], class(image_bundle));

%% radial kernel
for it = 1:n
    current_image = double(image_bundle(:,:,it));

    % Prepare data
    [nx, ny] = size(current_image);
    
    p1 = fsup * nx;
    p2 = fsup * ny;
    sigma = 0.8* fsup;
    S1a = psfmat(p1, sigma);
    S2a = psfmat(p2, sigma);
    S1 = kron(eye(nx), ones(1, fsup)) * S1a;
    S2 = kron(eye(ny), ones(1, fsup)) * S2a;

    % Initialize
    U = S1' * current_image * S2;
    G1 = S1' * S1;
    G2 = S2' * S2;
    
    % Compute Zoom Image
    [current_image, nit] = CG2(U, G1, G2, kappa, lambda, delta);
    
    current_image = cast(current_image, class(image_bundle));
    enhanced_bundle(:,:,it) = current_image;
end
end

% Turbo Zoom Functions
function Z = asypsp2(Y, p, pars)
    W = 0 * Y + 0.5;
    for it = 1:10
       Z = turbopsp(Y, pars, W);
       R = Y - Z;
       Wold = W;
       W = p * (R > 0) + (1 - p) * (R <= 0);
       sw = sum(W(:) ~= Wold(:));
       if sw == 0
         break
       end
    end
end

function B = bsplbase(x, bpars)
    % Fast computation of a B-spline basis,
    % of degree "deg", at positions "x",
    % on a uniform grid with "ndx" intervals between "x0" and "x1".
    % Saver computations
    %
    % Paul Eilers, 1996
    
    x0 = bpars(1); x1 = bpars(2); ndx = bpars(3); deg = bpars(4);
    x = x(:);
    if (min(x) < x0) | (max(x) > x1)
      disp('Some elements of x out of bounds !!')
      return
    end
    dx = (x1 - x0) / ndx;
    t = x0 + dx * ((-deg):(ndx-1));
    T = ones(size(x)) * t;
    X = x * ones(size(t));
    D = (X - T) / dx;
    B = diff([zeros(size(x)), D <= 1]')';
    r = [2:length(t) 1];
    for k = 1:deg
      B = (D .* B + (k + 1 - D) .* B(:, r)) / k;
    end
end

function [X, it] = CG2(U, G1, G2, kappa, lambda, delta)

    R = U;
    P = R;
    n1 = size(G1, 2);
    n2 = size(G2, 2);
    X = zeros(n1, n2);
    D1 = diff(eye(n1));
    D2 = diff(eye(n2));
    V1 = lambda * D1' * D1;
    V2 = lambda * D2' * D2;
    
    for it = 1:100
      Q = G1 * P * G2 + kappa * P + V1 * P + P * V2';
      alpha = sum(R(:) .^ 2) / sum(P(:) .* Q(:));
      X = X + alpha * P;
      Rnew = R - alpha * Q;
      rs1 = sum(R(:) .^ 2);
      rs2 = sum(Rnew(:) .^ 2);
      beta = rs2 / rs1;
      P = Rnew + beta * P;
      R = Rnew;
      rms = sqrt(rs1 / (n1 * n2)); 
%       disp([it log10(rms)])
      if rms < delta
        break
      end
    end
end

function S = psfmat(n, sigma)
    % Create a point spread matrix with width sigma and length n
    
    u = -n:n;
    v = exp(-(u' / sigma) .^ 2 / 2);
    v = v / sum(v);
    S = zeros(n, n);
    nv = length(v);
    for i = 1:n
        l = max(i - n, 1);
        u = min(i + n, n);
        l2 = max(l - i + n + 1, 1);
        u2 = min(u - i + n + 1, nv);
        S(i, l : u) = v(l2 : u2);
    end
end
    
function C = rowtens(A, B)
    % Compute (flattened) tensor products per row of A and B
    na = size(A, 2);
    nb = size(B, 2);
    ea = ones(1, na);
    eb = ones(1, nb);
    C = kron(A, eb) .* kron(ea, B);
end

function [F, A, Bc, Br] = turbopsp(Z, Ppars, W);
    % Very fast fitting of tensor P-splines to a full data matrix Z
    % Weights in W
    % Row 1 of Ppars: [nseg bdeg lambda pord] for columns
    % Row 2 of Ppars: dito for rows
    
    [m n] = size(Z);
    if nargin < 3
        W = ones(m, n);
    end
    
    % Prepare basis and penalty for columns
    cpars = [0 n+1 Ppars(2, :)];
    Bc = bsplbase((1:n)', cpars);
    nc = size(Bc, 2);
    Dc = diff(eye(nc), cpars(6));
    Pc = cpars(5) * Dc' * Dc;
    
    % Prepare basis and penalty for rows
    rpars = [0 m+1 Ppars(1, :)];
    Br = bsplbase((1:m)', rpars);
    nr = size(Br, 2);
    Dr = diff(eye(nr), rpars(6));
    Pr = rpars(5) * Dr' * Dr;
    
    % Do the fitting
    A = turbotens(Z, Bc, Br, Pc, Pr, W);
    F = Br * A * Bc';
end

function A = psp2turbo(Z, Bc, Br, Pr, Pc, W)
    % Very fast fitting of tensor P-splines to a full data matrix Z
    % Weights in W
    % Basis and penalty matrix for columns in Bc and Pc
    % Basis and penalty matrix for rows in Br and Pr
    
    [m n] = size(Z);
    if nargin < 6
        W = ones(m, n);
    end
    
    nc = size(Bc, 2);
    nr = size(Br, 2);
    T = rowtens(Br, Br)' * W * rowtens(Bc, Bc);
    T = reshape(T, [nr nr nc nc]);
    T = permute(T, [1 3 2 4]);
    T = reshape(T, nr * nc, nr * nc) + kron(Pc, eye(nr)) + kron(eye(nc), Pr);
    s = reshape(Br' * (W .* Z) * Bc, nr * nc, 1);
    A = reshape(T \ s,  nr, nc);
end        