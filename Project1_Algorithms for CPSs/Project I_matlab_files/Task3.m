clear all
close all
clc

%% Dati
q = 6;  % sensors
p = 7;  % cells

D = -[46 58 76 69 67 80 61;
      54 49 72 63 56 65 59;
      52 50 56 58 58 62 42;
      61 62 49 61 60 65 44;
      73 78 65 69 55 57 61;
      72 65 69 47 53 44 62];

y= -[62 58 41 46 64 63]';

epsilon = 1/5;
D_norm = normalize(D);
G = [D_norm eye(q)];
G = normalize(G);
lambda = 1 * ones(q+p,1);


%% PARAMETRI LASSO
eps = 10^-8;
tau = norm(D_norm, 2).^-2 - eps;
delta = 10^-12;
n_attacks = 1;

%% LASSO 
for h = 1:q
     
    % Creazione del vettore di misurazione attaccato
    y_attack = y;
    y_attack(h) = y_attack(h) + epsilon*y_attack(h);

     % Risoluzione del problema di LASSO
 
    w_t = zeros(p+q, 1);

    while true
        grad = (G')*(y_attack-G*w_t);
        arg = w_t + tau*grad;
        l = tau*lambda;
        w_t_next = sign(arg).*max(abs(arg) - l, 0);

        if norm(w_t_next - w_t, 2) < delta % Tmax
            break;
        end

        % update x_t
        w_t = w_t_next;

    end

    y_attack
    w_t
     % Identificazione delle celle in cui si trovano i target presenti
    target_indices_attack = find(w_t(1:p) ~= 0);
    num_targets_attack = length(target_indices_attack);

     % Stampa dei risultati
    fprintf('Attacco al sensore %d\n', h);
    fprintf('Numero di target presenti: %d\n', num_targets_attack);
    fprintf('Posizione dei target presenti: %s\n\n', mat2str(target_indices_attack));
  
end

