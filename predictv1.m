%Função que executa o predição da Rede Neural
function [saida_validacao_Y, count_acertos, count_erros, acuracia, mse] = predictv1( ...
    Whi, bias_hi, Woh, bias_oh, k, flag_fa, dados, dados_y)

    %Função de ativação
    syms x
    sigmoide(x) = 1./(1+exp(-x));
    tanh(x) = (1-exp(-2*x))/(1 + exp(-2*x));

    count_acertos = 0;
    count_erros = 0;
    
    size_base = size(dados,2);
    e_mse = 0;
    for indice_exemplo = 1:size_base
        
        fprintf('%.2f.',indice_exemplo/size_base)
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %2. Calcular entrada da camada escondida%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        net_h = Whi * dados(:,indice_exemplo) + bias_hi * ones(1,size(dados(:,indice_exemplo),2));
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %3. Calcular a saída da camada escondida%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if flag_fa == 1
            Yh = double(sigmoide(net_h)); % Função de Ativação: Sigmóide
        else
            Yh = double(tanh(net_h)); %Função de Ativação: Tanh
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %4. Calcular entrada da camada de saída%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        net_o = Woh * Yh + bias_oh * ones(1,size(Yh,2));
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %5. Calcular a saída da rede neural%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Y = k * net_o;
        
        %{
        if (dados_y(:,indice_exemplo) == round(Y))
            count_acertos=count_acertos +1;
        else
            count_erros=count_erros+1;
        end
        %}
        %%%%%%%%%%%%%%%%%%%%%%%%%%%
        %6. Calcular erro de saída%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%
        E = dados_y(:,indice_exemplo) - Y;
        if size(E,1) > 1 
             e_mse = e_mse + E.^2;
        else
             e_mse = e_mse + E^2;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %7. Valores validados              %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        saida_validacao_Y(:,indice_exemplo) = Y;    
        fprintf('\b\b\b\b\b')
        %fprintf('Exemplo: %d; Saída Esperada: %d / %s; Saída Prevista: %.2f / %s.\n', ...
        %        indice_exemplo,data_Y_teste(:,indice_exemplo),getAlvo(data_Y_teste(:,indice_exemplo)),Y,getAlvo(Y));
    end
    fprintf('\t100%%')
    %Registra o Erro Quadrático Médio e a Acurácia do Teste a cada iteração
    unificaMSE = mean(e_mse(:));
    mse = unificaMSE/size_base;
    acuracia = count_acertos/size_base*100;
end