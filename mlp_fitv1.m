%Função que executa o aprendizado da Rede Neural
function [Whi, bias_hi, Woh, bias_oh, treinamento_e_mse, evo_mse, dados_Erro_r2] = mlp_fitv1( ...
    cam_entrada_tam,cam_oculta_tam,cam_saida_tam, ...
    epocas,k,eta,flag_fa, rnaSeguinte, dados_treino, dados_Erro_r1)

    %Função de ativação
    syms x
    sigmoide(x) = 1./(1+exp(-x));
    tanh(x) = (1-exp(-2*x))/(1 + exp(-2*x));

    evo_mse = [];

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 1. Inicializar pesos das camadas%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

       Whi = rand(cam_oculta_tam,cam_entrada_tam) - 0.5;
       % bias da saída da camada de entrada para a camada oculta
       bias_hi = rand(cam_oculta_tam,1) - 0.5;    
    
    if rnaSeguinte == 1 
       % Inicializar pesos da saída da camada oculta para a camada de saída
       % quando é primeira rna
       Woh = rand(cam_saida_tam,cam_oculta_tam) - 0.5;
       % bias da saída da camada oculta para a camada de saída
       bias_oh = rand(cam_saida_tam,1) - 0.5;
    elseif rnaSeguinte == 2
       % Inicializar pesos da saída da camada oculta para a camada de saída
       % quando não é primeira rna
       Woh = zeros(cam_saida_tam, cam_oculta_tam);
       bias_oh = zeros(cam_saida_tam, 1);
    end


    dados_Erro_r2 = zeros(1,size(dados_Erro_r1,2));

    fprintf(['\nParâmetros do Modelo: \n Input Layer: %d \n Hidden Layer: %d \n' ...
        ' Output Layer: %d \n Função de Ativação: %d (1- Sigmoide; 2- Tanh) \n ' ...
        'Taxa de Aprendizado: %.3f \n K da Função Linear: %d \n Épocas: %d\n'], ...
        cam_entrada_tam,cam_oculta_tam,cam_saida_tam,flag_fa,eta,k,epocas)


    size_base = size(dados_treino,2);
    fprintf('\nBase de Dados de Treino: %d exemplos \n',size_base)
    
    fprintf('\nBLOCO DE TREINAMENTO')

    % Erro médio quadrático das épocas durante o Treinamento
    treinamento_e_mse = 0;
    
    
    for loop = 1:epocas
        
        fprintf('\n Época: %d ->  \b',loop);
        
        
        % Erro médio quadrático de cada época durante o Treinamento
        epoca_e_mse = 0;
        epoca_n_mse = 0;
        
        for indice_exemplo = 1:size_base
    
            fprintf('%.2f.',indice_exemplo/size_base)

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %2. Calcular entrada da camada escondida%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            net_h = Whi * dados_treino(:,indice_exemplo) + bias_hi*ones(1,size(dados_treino(:,indice_exemplo),2));
    
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
            net_o = Woh*Yh + bias_oh*ones(1,size(Yh,2));
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %5. Calcular a saída da rede neural%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            Y = k*net_o;
    
            %%%%%%%%%%%%%%%%%%%%%%%%%%%
            %6. Calcular erro de saída%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%
            E = dados_Erro_r1(:,indice_exemplo) - Y;
            dados_Erro_r2(:,indice_exemplo) = E;
            if size(E,1) > 1 
                epoca_e_mse = epoca_e_mse + E.^2;
            else
                epoca_e_mse = epoca_e_mse + E^2;
            end
            epoca_n_mse = epoca_n_mse + 1;
    
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %7. Calcular variação dos pesos entre as camadas de saída e escondida%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            df_oh = k*ones(size(net_o));
            delta_bias_oh = eta*sum((E.*df_oh),1)';
            delta_Woh = eta*(E.*df_oh)*Yh';
    
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %8. Calcular erro retropropagado%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            Eh = -Woh'*(E.*df_oh);
    
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %9. Calcular variação dos pesos entre as camadas escondida e de entrada%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if flag_fa == 1
                df_hi = Yh - Yh.^2; % Função de Ativação: Sigmóide
            else
                df_hi = 1 - (Yh.^2); %Função de Ativação: Tanh
            end
    
            delta_bias_hi = -eta * sum((Eh.*df_hi),1)';
            delta_Whi = -eta*(Eh.*df_hi)*dados_treino(:,indice_exemplo)';
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %10.Calcular novos valores dos pesos%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            Whi = Whi + delta_Whi;
            bias_hi = bias_hi + delta_bias_hi;
            
            Woh = Woh + delta_Woh;
            bias_oh = bias_oh + delta_bias_oh;
            
            fprintf('\b\b\b\b\b')
    
            %fprintf('Época: %d; Exemplo: %d; Saída Esperada: %d / %s; Saída Prevista: %.2f / %s; Erro: %.2f.\n', ...
            %    loop,indice_exemplo,data_Y_train(:,indice_exemplo),getAlvo(data_Y_train(:,indice_exemplo)),Y,getAlvo(Y),E);
            
            
        end 
        unificaMSE = mean(epoca_e_mse(:));
        treinamento_e_mse = treinamento_e_mse + unificaMSE/epoca_n_mse;
        fprintf('100%% --> MSE: %.2f',unificaMSE/epoca_n_mse)
        evo_mse(end+1) = unificaMSE/epoca_n_mse;


    end   
    treinamento_e_mse = treinamento_e_mse/epocas;
end