

# %%
# Step 1: Import necessary libraries
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
from tqdm import tqdm  # tqdm helps to display a progress bar during the loop

# Step 2: Load the pre-trained GPT-2 model and tokenizer
# "gpt2" is the small version, but you can use "gpt2-medium", "gpt2-large", or "gpt2-xl" for larger models
model_id = "gpt2"  
model = GPT2LMHeadModel.from_pretrained(model_id).to("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

# %%

# Step 3: Define the chunk of text that we want to compute perplexity for
# This text will be tokenized and passed through the model
text = """
'den de mai om aftenen, omtrent kl. , udbrød jld i kjøbmand salomon simonsens sted i sæb med saadan voldsomhed, at inden sprøiterne kom til brandstedet stod hele bygningen i lys lue, 
og kort derpaa blev skomager mads kroghssted angrebet, hvilket ved brandcorpsets raske og utrættelige bestræbelser dog adskillige gange blev dæmpet; 
men med et udbrød jlden paany i kjøbmand h. schioldans ladebygning, og, da der ikke i dette øieblik syntes nogen fare mere for kroghs sted, 
blev corpset commanderet derhen; men her var ingen redning at bevirke, da laden var halv fuld af straa; 
ved denne formerede branden sig til feldbereder wetterdoms og derfra til bertheljensen knudens sted, 
der alle afbrændte til grunden til lykke kom jngen til skade derved, og alle fik deres jndbo paa lidet nær reddet fra luerne naar undtages kjøbmand simonsen, 
der næsten jntet fik reddet af jndboet og aldeles ingen af sine kjøbmandsvarer, saavelsom foldbereder wetterdom, 
der var fraværende, og mistede en betydelig deel af sit bohave; men til lykke for disse tvende mænd var deres jndbo og varelager assureret. 
da der nu intet meere kunde udrettes ved de trende sidste steder, bleve sprøitemmeratter flyttede op til krogbs sted, der nu igjen var angrebet; 
men nu var inen redning mere mulig; thi jlden havde taget saa voldsomt om sig, at al anstrængelse var forgjeves. 
et tille sted, tilhørende peder langtved, blev og angrebet; men her maatte jlden døie sig for magten, 
da nogle raske mænd uforfærdet stege op paa huset og nedrev endeel tag, som luerne havde angrebet. 
de vare alle straatækkede. kl omtrent næste morgen var faren forbi. i aalb. av.) 
"""

text2 = """
graadig faldt jeg hen over handsken, trykte tusinde kysse derpaa; thi hvem kunde den vel tilhøre uden den indtagende naboerske. 
de røde viinpletter tilkjendegav det noksom: jeg havde i aabenbar forlegenhed stukken den til mig. 
med den engelske fortskaftede trefork maatte det ved have samme sammenhæng. 
jeg klædte mig paa og maaske med mere ombyggelighed end nogensinde før. 
den arme johan som presiderede ved tøilettet, blev igjendygtig udskjældt; thi han opførte sig ligesaa ubehændig som langsomt. 
da halsløifen to gange mislykkedes, bad jeg ham gaae fanden i vold og fuldendte selv det konstige værk. 
j det jeg lagde frakken i sine tilbørlige folder, kastede jeg et selvtilfreds blik paa mit adoniserede billede i speilet, 
et andet ud gjennem vinduet og see engang, der vandrede det hulde barn i sneehvid morgennegligs, omgivet af gaarsdagens selskab paa den saakaldte kamp, eller indhegnede eng. 
j to spring var jeg nede hos hende. da jeg traaede ud af døren, mødte hendes blik mig, hun stødte paa edward som ledsagede hende, og talte nogle ord til hendes blaae veninde, som uden tvivl angik mig. 
jeg rettede mine skridt mod indgangen af promenaden, ueenimed mig selv, om jeg skulle tale til de fremmedefor at undskylde min ubehændighed i gaar; 
men mit bryst var saa beklemt og jeg følte tydelig at jeg ikke kunde frembringe et eneste ord. 
for at faae tid til at fatte mig noget, ville jeg just bøie af til venstre, da hr. edward traaede hen til mig og bød mig en venlig god morgen. 
tager jeg ikke feil, sagde han, saa var de i gaaraftes mine damers naboe ved bordet. 
ja jeg var saa ulykkelig. - den lille havde just kastet sine øine paa mig og indviklede mig derved i en labyrinth af lykke og ulykke, som jeg aldrig kunde finde ud af.
"""

# print len of each text
print(f"Length of text1: {len(text)}")
print(f"Length of text2: {len(text2)}")

# remove linebreaks from text
text = text.replace("\n", " ")
text2 = text2.replace("\n", " ")

# Step 4: Tokenize the text using the tokenizer from Hugging Face
# This converts the text into tokens (numbers that the model understands)
encodings = tokenizer(text, return_tensors="pt")

# Step 5: Set parameters for sliding window
max_length = model.config.n_positions  # maximum length the model can process (default 1024 for GPT-2)
stride = 512  # Overlap between chunks
seq_len = encodings.input_ids.size(1)  # The total number of tokens in our text (after tokenization)

# Step 6: Initialize variables to accumulate loss and count tokens
nll_sum = 0.0  # Sum of negative log-likelihoods, which we will average later to get the perplexity
n_tokens = 0  # Count the valid tokens that we process
prev_end_loc = 0  # Keep track of where we were in the text

# Step 7: Process the text in chunks using a sliding window (step size of 'stride')
for begin_loc in tqdm(range(0, seq_len, stride)):  # creates indices in steps of 'stride'
    end_loc = min(begin_loc + max_length, seq_len)  # Define the end location
    trg_len = end_loc - prev_end_loc  # Length of the target sequence (last chunk might be smaller than 'stride')

    # Step 8: Slice the input tensor to get the chunk we are working on
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Step 9: Create a target tensor (copy of input_ids) and set all tokens except the target ones to -100
    target_ids = input_ids.clone()  # Copy the input tokens
    target_ids[:, :-trg_len] = -100  # Set the tokens that are part of the context (not target tokens) to -100
    
    # Step 10: Pass the input through the model to compute the loss
    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)  # Forward pass
        neg_log_likelihood = outputs.loss  # The loss is the negative log-likelihood (NLL)

    # Step 11: Accumulate the total NLL (negative log-likelihood) and valid token count
    # We only want to count the tokens that are valid (not the context tokens set to -100)
    num_valid_tokens = (target_ids != -100).sum().item()
    nll_sum += neg_log_likelihood.item() * num_valid_tokens  # Add the NLL for this chunk, weighted by the valid tokens
    n_tokens += num_valid_tokens  # Add the number of valid tokens for this chunk to the total token count

    prev_end_loc = end_loc  # Move to the end of the current chunk
    if end_loc == seq_len:  # If we reach the end of the text, stop
        break

# Step 12: Compute the average negative log-likelihood (NLL) across all tokens
avg_nll = nll_sum / n_tokens # avg nll / token

# Step 13: Convert the average NLL to perplexity by exponentiating it
# Perplexity is just the exponent of the average NLL
ppl = torch.exp(torch.tensor(avg_nll))

print(f"Perplexity: {ppl.item()}")
# %%
