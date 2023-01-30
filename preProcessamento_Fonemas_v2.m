% ---------- Inicialização ----------

tam_treino = 0.7; % Tamanho do pacote de treino
tam_valida = 0.3; % Tamanho do pacote de validação
tam_teste = 0.0; % Tamanho do pacote de teste

% Este arquivo é executado de modo a transformar os arquivos de áudio em um
% dataset estruturado, aplicando compactação por média.
padraoTam = 160; % Quantidade de atributos requeridas para cada padrão de entrada
fonemasTotal = 6; % Quantidade de fonemas/classes
padraoPorFonema = 320; % Quantidade de padrões para cada fonema
plotAmostras = 1; % Define se a intâncias serão exibidas e salvas em forma gráfica após o tratamento.


[X, Y] = carregaArquivos(padraoTam, (fonemasTotal * padraoPorFonema), plotAmostras);
[X_treino, X_valida, X_teste] = preparaBases(X, tam_treino, tam_valida, tam_teste);
save("preProc_v2.mat");
fprintf('\nTratamento dos arquivos de áudio finalizado');

% Funcao carregaArquivos
% Adaptado a partir de https://github.com/jpac1207/Neural_Network_Matlab
% Le os arquivos nos diretorios indicados
% Aplica a FFT
% Adiciona o vetor de referencia

function[X, Y] = carregaArquivos(padraoTam, amostrasTotal, plotAmostras)
    dirraiz = 'C:\Users\oficial\Documents\DCCMAPI\Atv05\fonemas\';
    dir1 = strcat(dirraiz, 'di\');
    dir2 = strcat(dirraiz, 'rei\');
    dir3 = strcat(dirraiz, 'ta\');
    dir4 = strcat(dirraiz, 'es\');
    dir5 = strcat(dirraiz, 'quer\');
    dir6 = strcat(dirraiz, 'da\');
    fonemas.classMap = containers.Map({dir1, dir2, dir3, ...
        dir4, dir5, dir6}, {1, 2, 3, 4, 5, 6}); 
    X = zeros(amostrasTotal, padraoTam);
    Y = zeros(amostrasTotal,1);
    amostrasContador = 1;
    % Itera sobre todas as pastas
    for chave = keys(fonemas.classMap)        
        arquivosDiretorio = dir(chave{1});
        arquivosTotal = size(arquivosDiretorio, 1); 
        % Itera sobre todos os arquivos na pasta atual
        for i=1:arquivosTotal
            arq = arquivosDiretorio(i);
            % Se o elemento atual for um arquivo
            if ~arq.isdir
                arqEmProcesso = strcat(arq.folder, '\', arq.name);
                amostra = audioread(arqEmProcesso);
                amostrafft = abs(fft(amostra));
                amostrafftHalf = floor(size(amostrafft, 1)/2);                 
                amostrafft = amostrafft(1:amostrafftHalf, 1); % Utiliza apenas metade dos valores                  
                groupSize = floor(size(amostrafft, 1)/padraoTam);                
                padraoEntrada = zipEntrada(amostrafft, padraoTam, groupSize); 
                if plotAmostras 
                    plot(padraoEntrada);
                    hold on;                
                end
                X(amostrasContador, :) = padraoEntrada(1, :);               
                rotulo =  fonemas.classMap(chave{1});                
                Y(amostrasContador,1) = rotulo;
                amostrasContador = amostrasContador + 1;
            end
        end
        if plotAmostras 
            hold off;
            fonemaNome = split(chave, '\');       
            title(['Padrões de Entrada Fonema: '  fonemaNome{8, 1}]);
            xlabel('Grupo');
            ylabel('Média das Amplitudes do Grupo');
            saveas(gcf, ['treinofg\' 'Padrões_de_Entrada_Fonema_'  fonemaNome{8, 1} '.png'])
        end
    end
    X = [X Y];
end

% Funcao zipEntrada
% Adaptado a partir de https://github.com/jpac1207/Neural_Network_Matlab
% Realiza a compactação da entrada, retornando um vetor de
% 'grupoQuant' colunas atráves das médias do grupos formados por
% 'grupoTamanho' elementos
function[X_saida] = zipEntrada(entrada, grupoQuant, grupoTamanho)    
    X_saida = zeros(1, grupoQuant);      
    startPosition = 1;
    % Para cada atributo no padrão de saída, cálcula a média do padrão
    % de entrada
    for j=1:grupoQuant        
        slice = entrada(startPosition:(startPosition + grupoTamanho - 1));
        X_saida(1, j) = mean(slice);
        startPosition = startPosition + grupoTamanho;
    end  
end


% Funcao preparaBases
% Adaptado a partir do algoritmo de Alisson EGMendonca
% Divide as bases em 3 conforme a proporcao indicada pelo parametros
function [treino,valida,teste] = preparaBases(X, tam_treino, tam_valida, tam_teste)
    matrix = X;
    data_size = size(matrix);
    %cria o índice para cada linha do arquivo
    indexes = linspace(1,data_size(1),data_size(1));
    %mistura os índices para dispor as linhas de forma aleatória no
    %particionamento
    indexes = indexes(randperm(length(indexes)));
    
    %particiona a base de Treino
    i_treino = round(length(indexes)*tam_treino);
    indexes_treino = indexes(1:i_treino);
    treino = matrix(indexes_treino,:);
    
    %particiona a base de Validação
    i_validacao = round(length(indexes) * tam_valida) + i_treino;
    indexes_valida = indexes(i_treino+1:i_validacao);
    valida = matrix(indexes_valida,:);
    
    %particiona a base de Teste
    i_teste = round(length(indexes) * tam_teste) + i_validacao;
    if(i_teste > data_size(1))
        i_teste = data_size(1);
    end
    indexes_teste = indexes(i_validacao+1:i_teste);
    teste = matrix(indexes_teste,:);
end
