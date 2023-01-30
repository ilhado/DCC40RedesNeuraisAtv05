% --- Implementação do algoritmo de RNA via MLP 
% ---- funções adicionais utilizadas mlp_fit.m, predict.m e grafico.m
% ---------- Parâmetros Gerais ----------
% clc;
clear variables;
epocas = 50; % Define a quantidade de épocas do treinamento
eta = 0.01; % Learning Rateos para computar as médias.
k = 1; %Constante da Função de Ativação Linear
flag_fa_r1 = 1; % Seleção da Função de Ativação 1 - Sigmoide; 2 - Tanh
flag_fa_r2 = 2; % Seleção da Função de Ativação 1 - Sigmoide; 2 - Tanh
rnaSeguinte = 1; % Define se a rna não é a primeira 1 - primeira; 2 - seguintes

load('C:\Users\oficial\Documents\DCCMAPI\Atv05\preProc_v2.mat','X_treino', 'X_valida','X_teste', 'X', 'Y');
tamSaida = size(X,2) - 1;
repeticoes = 10;

%Inicialização dos Arrays que armazenarão os resultados da Predição de
%Validação
acuracia_validacao = zeros(1,repeticoes);
mse_validacao = zeros(1,repeticoes);
count_acertos_validacao = zeros(1,repeticoes);
count_erros_validacao = zeros(1,repeticoes);

%Inicialização dos Arrays que armazenarão os resultados da Predição de
%Teste
acuracia_teste = zeros(1,repeticoes);
mse_teste = zeros(1,repeticoes);
count_acertos_teste = zeros(1,repeticoes);
count_erros_teste = zeros(1,repeticoes);

% Define os dados preditivos e os alvos para o Treinamento
dados_treino = X_treino(:,1:tamSaida)';
dados_Y_treino = X_treino(:,tamSaida+1:tamSaida+1)';

% Define os dados preditivos e os alvos para a Validação
dados_valida = X_valida(:,1:tamSaida)';
dados_Y_valida = X_valida(:,tamSaida+1:tamSaida+1)';

% Define os dados preditivos e os alvos para o Teste
dados_teste = X_teste(:,1:tamSaida)';
dados_Y_teste = X_teste(:,tamSaida+1:tamSaida+1)';

%Quantidade de Neurônios das Camadas
camada_entrada_tam = size(dados_treino,1);
camada_saida_tam = 1;
camada_oculta_tam = 12;
%camada_oculta_tam = 2*size(dados_treino,1);

cada_mse = [];

for atividade = 1:repeticoes
    
    fprintf('\nATIVIDADE 01 - EXECUÇÃO %d de %d...\n',atividade,repeticoes)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % BLOCO DE TREINAMENTO DO MODELO %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    
    
    fprintf('\n Treinamento RNA via MLP - ativação sigm\n');
    [Whi_r1, bias_hi_r1, Woh_r1, bias_oh_r1, treinamento_e_mse_r1, evo_mse, dados_Erro_r1] = mlp_fitv1(camada_entrada_tam, camada_oculta_tam, camada_saida_tam, ...
        epocas, k, eta, flag_fa_r1, rnaSeguinte, dados_treino, dados_Y_treino);
    fprintf('\n MSE Treinamento: %.2f\n',treinamento_e_mse_r1);
    cada_mse = [cada_mse evo_mse];

    rnaSeguinte = 2;
    fprintf('\n Treinamento RNA via MPL - ativação tanh\n');
    [Whi_r2, bias_hi_r2, Woh_r2, bias_oh_r2, treinamento_e_mse_r2, evo_mse, dados_Erro_r2] = mlp_fitv1(camada_entrada_tam, camada_oculta_tam, camada_saida_tam, ...
        epocas, k, eta, flag_fa_r2, rnaSeguinte, dados_treino, dados_Erro_r1);
    fprintf('\n MSE Treinamento: %.2f\n',treinamento_e_mse_r2);
    cada_mse = [cada_mse evo_mse];
%{    
    fprintf('\n\nBLOCO DE VALIDAÇÃO da MLP tanH->  \b')
    [count_acertos,count_erros,acuracia,mse] = predict( ...
        Whi, bias_hi, Woh, bias_oh, k, flag_fa, dados_valida, dados_Y_valida);
    fprintf('\n RESULTADOS:\n');
    fprintf('  Total: %d; Acertos: %d; Erros: %d; Acurácia: %.2f%%; MSE: %.2f.\n', ...
        size(dados_valida,2),count_acertos,count_erros, acuracia, mse);

    flag_fa = 1;
    fprintf('\n Treinamento RNA via MPL - ativação sigmóide\n');
    [Whi, bias_hi, Woh, bias_oh, treinamento_e_mse, evo_mse, dados_Erro_r3] = mlp_fitv1(camada_entrada_tam, camada_oculta_tam, camada_saida_tam, ...
        epocas, k, eta, flag_fa, dados_treino, dados_Erro_r2);
    fprintf('\n MSE Treinamento: %.2f\n',treinamento_e_mse);
    cada_mse = [cada_mse evo_mse];    
%}    

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % BLOCO DE VALIDAÇÃO DO MODELO %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    fprintf('\n\nBLOCO DE VALIDAÇÃO E TESTE ->  \b')
    [saida_validacao_r1,count_acertos_r1,count_erros_r1,acuracia_r1,mse_r1] = predictv1( ...
        Whi_r1, bias_hi_r1, Woh_r1, bias_oh_r1, k, flag_fa_r1, dados_valida, dados_Y_valida);

    [saida_validacao_r2, count_acertos_r2,count_erros_r2,acuracia_r2,mse_r2] = predictv1( ...
        Whi_r2, bias_hi_r2, Woh_r2, bias_oh_r2, k, flag_fa_r2, dados_valida, dados_Y_valida);    


    tam_base = size(dados_Y_valida,2);
    acertos = 0;
    erros = 0;
    for indice = 1:tam_base
        saida = saida_validacao_r1(:,indice) + saida_validacao_r2(:,indice);
        if (dados_Y_valida(:,indice) == round(saida))
            acertos = acertos + 1;
        else
            erros = erros + 1;
        end
    end

    acuracia = acertos / tam_base * 100;
    mse = (mse_r1 + mse_r2)/2;
    fprintf('\n RESULTADOS:\n');
    fprintf('  Total: %d; Acertos: %d; Erros: %d; Acurácia: %.2f%%; MSE: %.2f.\n', ...
        size(dados_valida,2),acertos,erros, acuracia, mse);
    

    acuracia_validacao(atividade) = acuracia;
    mse_validacao(atividade) = mse;
    count_acertos_validacao(atividade) = acertos;
    count_erros_validacao(atividade) = erros;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % BLOCO DE TESTE DO MODELO %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %{
    fprintf('\n\nBLOCO DE TESTES ->  \b')

    [saida_teste_r1,count_acertos_r1,count_erros_r1,acuracia_r1,mse_r1] = predictv1( ...
        Whi_r1, bias_hi_r1, Woh_r1, bias_oh_r1, k, flag_fa_r1, dados_teste, dados_Y_teste);

    [saida_teste_r2, count_acertos_r2,count_erros_r2,acuracia_r2,mse_r2] = predictv1( ...
        Whi_r2, bias_hi_r2, Woh_r2, bias_oh_r2, k, flag_fa_r2, dados_teste, dados_Y_teste);    


    tam_base_teste = size(dados_Y_teste,2);
    acertos_teste = 0;
    erros_teste = 0;
    for indice = 1:tam_base_teste
        saida_teste = saida_teste_r1(:,indice) + saida_teste_r2(:,indice);
        if (dados_Y_teste(:,indice) == round(saida_teste))
            acertos_teste = acertos_teste + 1;
        else
            erros_teste = erros_teste + 1;
        end
    end

    acuracia_t = acertos_teste / tam_base_teste * 100; 
    mse_t = (mse_r1 + mse_r2)/2;
    fprintf('\n RESULTADOS:\n');
    fprintf('  Total: %d; Acertos: %d; Erros: %d; Acurácia: %.2f%%; MSE: %.2f.\n', ...
        size(dados_teste,2),acertos_teste,erros_teste, acuracia_t, mse_t);
    

    acuracia_teste(atividade) = acuracia_t;
    mse_teste(atividade) = mse_t;
    count_acertos_teste(atividade) = acertos_teste;
    count_erros_teste(atividade) = erros_teste;
    %}

end


 %Geração do Gráfico que compara o MSE do treinamento 
 todasEpocas = size(cada_mse,2);
 grafico(1:todasEpocas, cada_mse, "Resultados do MSE",[],"",[],"",[],"","Épocas","MSE", "Variação do MSE por épocas");
    

%Geração do Gráfico que compara as acurácias da predição para as bases de
%Validação e de Teste
grafico(1:repeticoes,acuracia_validacao,"Acurácia da Validação e teste", ...
    [],"",[],"",[],"","Nº da Execução","Acurácia (%)", "Acurácia por treinamento - 50 épocas por treinamento")

%Geração do Gráfico que compara o MSE da predição para as bases de
%Validação e de Teste
grafico(1:repeticoes,mse_validacao,"MSE da Validação e Teste", ...
    [],"",[],"",[],"","Nº da Execução","MSE", "MSE médio por treinamento - 50 épocas por treinamento")

%Geração do Gráfico que compara os Acertos e Erros da predição para as bases de
%Validação e de Teste
grafico(1:repeticoes,count_acertos_validacao,"Acertos (Validação)", ...
    count_erros_validacao,"Erros (Validação)", ...
    [],"", [],"", "Nº da Execução","Nº de Acertos/Erros", "Acertos vs Erros por treinamento")

 save('MelhorRede.mat','Whi_r2','bias_hi_r2','Woh_r2','bias_oh_r2');