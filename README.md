# Projeto de Detecção de Vivacidade

Este projeto tem como objetivo detectar a vivacidade de um rosto utilizando técnicas de reconhecimento facial e redes neurais. O desenvolvimento do projeto foi feito no arquivo `deteccao_liveness_notebook.ipynb`, com as dependências listadas em `requirements.txt`. Adicionalmente, os arquivos `deploy.prototxt.txt` e `res10_300x300_ssd_iter_140000.caffemodel` são necessários para o modelo de detecção facial. A aplicação web foi desenvolvida usando Streamlit e está no arquivo `streamlit-app.py`.

## Estrutura do Projeto

- `deteccao_liveness_notebook.ipynb`: Contém todo o desenvolvimento do projeto, incluindo o treinamento e a avaliação do modelo de detecção de vivacidade.
- `streamlit-app.py`: Código para a aplicação web utilizando Streamlit.
- `liveness.keras`: Modelo treinado após execução.
- `le.pickle`: Categorias treinadas para o modelo.
- `requirements.txt`: Lista de dependências necessárias para rodar o projeto.
- `deploy.prototxt.txt`: Configuração do modelo de rede neural para a detecção facial.
- `res10_300x300_ssd_iter_140000.caffemodel`: Pesos do modelo treinado para a detecção facial.
- `./videos`: Arquivos de vídeo para processamento.
- `./dataset`: Imagens processadas.

## Aplicação Web com Streamlit
A aplicação web desenvolvida em Streamlit permite que os usuários carreguem uma imagem e obtenham a detecção de vivacidade. A interface é simples e intuitiva, facilitando o uso da tecnologia por qualquer pessoa.
Clique [aqui](https://deteccaoliveness-bmnuzqisamtfhwo6d8iekn.streamlit.app/) para acessar a aplicação
