{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "2c146621",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Getting TFGPT2LMHeadModel from transformers\n",
    "from transformers import TFGPT2LMHeadModel\n",
    "\n",
    "# Getting GPT2Tokenizer from transformers\n",
    "from transformers import GPT2Tokenizer\n",
    "\n",
    "# Getting Tensorflow\n",
    "import tensorflow as tf"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9b175617",
   "metadata": {},
   "source": [
    "# Making Basic GPT2 Models for Text Generation Runs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "19906c44",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "All model checkpoint layers were used when initializing TFGPT2LMHeadModel.\n",
      "\n",
      "All the layers of TFGPT2LMHeadModel were initialized from the model checkpoint at gpt2-large.\n",
      "If your task is similar to the task the model of the checkpoint was trained on, you can already use TFGPT2LMHeadModel for predictions without further training.\n"
     ]
    }
   ],
   "source": [
    "# Getting Largest Models\n",
    "Large_Tokenizer = GPT2Tokenizer.from_pretrained(\"gpt2-large\")\n",
    "Large_Model = TFGPT2LMHeadModel.from_pretrained(\"gpt2-large\", pad_token_id = Large_Tokenizer.eos_token_id)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "5c47fa97",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"tfgp_t2lm_head_model\"\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "transformer (TFGPT2MainLayer multiple                  774030080 \n",
      "=================================================================\n",
      "Total params: 774,030,080\n",
      "Trainable params: 774,030,080\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "# Printing Summary\n",
    "Large_Model.summary()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "bbc55bab",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "50de27d70d7142a2a4a5b5be585e7679",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Downloading:   0%|          | 0.00/0.99M [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "dd3e7a373cb14ec996b41df8206bc2ba",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Downloading:   0%|          | 0.00/446k [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "66dd25795c804ff39824a5e46333e782",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Downloading:   0%|          | 0.00/718 [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "929c349c71de437cb591f95779e4200c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Downloading:   0%|          | 0.00/1.32G [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "All model checkpoint layers were used when initializing TFGPT2LMHeadModel.\n",
      "\n",
      "All the layers of TFGPT2LMHeadModel were initialized from the model checkpoint at gpt2-medium.\n",
      "If your task is similar to the task the model of the checkpoint was trained on, you can already use TFGPT2LMHeadModel for predictions without further training.\n"
     ]
    }
   ],
   "source": [
    "# Getting Medium Models\n",
    "Medium_Tokenizer = GPT2Tokenizer.from_pretrained(\"gpt2-medium\")\n",
    "Medium_Model = TFGPT2LMHeadModel.from_pretrained(\"gpt2-medium\", pad_token_id = Medium_Tokenizer.eos_token_id)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "f56562dc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"tfgp_t2lm_head_model_1\"\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "transformer (TFGPT2MainLayer multiple                  354823168 \n",
      "=================================================================\n",
      "Total params: 354,823,168\n",
      "Trainable params: 354,823,168\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "# Printing Summary\n",
    "Medium_Model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "16a9d9ab",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "da33c3775adc4ddbb1b0cfb778feb32e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Downloading:   0%|          | 0.00/0.99M [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3db210e259e14ace96cd0a3c4eefca99",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Downloading:   0%|          | 0.00/446k [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "00287c690ac7473f90980434443f2701",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Downloading:   0%|          | 0.00/665 [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "c48cb048a88d42e5a555380b940751af",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Downloading:   0%|          | 0.00/475M [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "All model checkpoint layers were used when initializing TFGPT2LMHeadModel.\n",
      "\n",
      "All the layers of TFGPT2LMHeadModel were initialized from the model checkpoint at gpt2.\n",
      "If your task is similar to the task the model of the checkpoint was trained on, you can already use TFGPT2LMHeadModel for predictions without further training.\n"
     ]
    }
   ],
   "source": [
    "# Getting Samll Models\n",
    "Small_Tokenizer = GPT2Tokenizer.from_pretrained(\"gpt2\")\n",
    "Small_Model = TFGPT2LMHeadModel.from_pretrained(\"gpt2\", pad_token_id = Small_Tokenizer.eos_token_id)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "b2dfe526",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"tfgp_t2lm_head_model_2\"\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "transformer (TFGPT2MainLayer multiple                  124439808 \n",
      "=================================================================\n",
      "Total params: 124,439,808\n",
      "Trainable params: 124,439,808\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "# Printing Summary\n",
    "Small_Model.summary()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "dd00da5e",
   "metadata": {},
   "source": [
    "\n",
    "# Trying - Greedy Search"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "47056f25",
   "metadata": {},
   "outputs": [],
   "source": [
    "# setting random seed\n",
    "Random_Seed_Value = 34\n",
    "tf.random.set_seed(Random_Seed_Value)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "a3047f58",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Setting MAximum Length of Output\n",
    "Maximum_Length = 100\n",
    "\n",
    "# Setting Input String for text Generation\n",
    "Input_Data = \"I don't know about you, but there's only one thing I want to do after a long day of work\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "679b59d4",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Encoding Input Text in all 3 models\n",
    "Large_Model_Input = Large_Tokenizer.encode(Input_Data, return_tensors='tf')\n",
    "Medium_Model_Input = Medium_Tokenizer.encode(Input_Data, return_tensors='tf')\n",
    "Small_Model_Input = Small_Tokenizer.encode(Input_Data, return_tensors='tf')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "eeb09ffd",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Finding Greedy Output\n",
    "Large_Model_Greedy_Output = Large_Model.generate(Large_Model_Input, max_length = Maximum_Length)\n",
    "Medium_Model_Greedy_Output = Medium_Model.generate(Medium_Model_Input, max_length = Maximum_Length)\n",
    "Small_Model_Greedy_Output = Small_Model.generate(Small_Model_Input, max_length = Maximum_Length)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "84103bd8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Output By Largest Model ->\n",
      "I don't know about you, but there's only one thing I want to do after a long day of work: go to the gym.\n",
      "\n",
      "I'm not talking about the gym that's right next to my house. I'm talking about the gym that's right next to my office.\n",
      "\n",
      "I'm not talking about the gym that's right next to my house. I'm talking about the gym that's right next to my office.\n",
      "\n",
      "I'm not talking about the gym\n"
     ]
    }
   ],
   "source": [
    "# Printing Output Large Model\n",
    "print(\"Output By Largest Model ->\")\n",
    "Large_Final_Greedy_Output = Large_Tokenizer.decode(Large_Model_Greedy_Output[0], skip_special_tokens = True)\n",
    "print(Large_Final_Greedy_Output)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "ed663961",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Output By Medium Model ->\n",
      "I don't know about you, but there's only one thing I want to do after a long day of work: eat.\n",
      "\n",
      "I'm not talking about the food I eat for breakfast, lunch, dinner, or even the food I eat for dinner. I'm talking about the food I eat for lunch.\n",
      "\n",
      "I'm talking about the food I eat for lunch.\n",
      "\n",
      "I'm talking about the food I eat for lunch.\n",
      "\n",
      "I'm talking about the food I eat\n"
     ]
    }
   ],
   "source": [
    "# Printing Output Medium Model\n",
    "print(\"Output By Medium Model ->\")\n",
    "Medium_Final_Greedy_Output = Medium_Tokenizer.decode(Medium_Model_Greedy_Output[0], skip_special_tokens = True)\n",
    "print(Medium_Final_Greedy_Output)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "140fe1f5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Output By Small Model ->\n",
      "I don't know about you, but there's only one thing I want to do after a long day of work. I want to get out of here. I want to get out of here. I want to get out of here. I want to get out of here. I want to get out of here. I want to get out of here. I want to get out of here. I want to get out of here. I want to get out of here. I want to get\n"
     ]
    }
   ],
   "source": [
    "# Printing Output Small Model\n",
    "print(\"Output By Small Model ->\")\n",
    "Small_Final_Greedy_Output = Small_Tokenizer.decode(Small_Model_Greedy_Output[0], skip_special_tokens = True)\n",
    "print(Small_Final_Greedy_Output)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d2b78f28",
   "metadata": {},
   "source": [
    "# Trying - Beam Search"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "0ebdf375",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0 I don't know about you, but there's only one thing I want to do after a long day of work, and that's to sit down and watch a movie.\"\n",
      "\n",
      "\"I know, I know,\" you say. \"But you're not going to like this one. It's not a good movie. I mean, it's a really bad movie, you know? And I'm not sure if you've ever seen it before, so I'll just tell you what it is\n",
      "\n",
      "1 I don't know about you, but there's only one thing I want to do after a long day of work, and that's to sit down and watch a movie.\"\n",
      "\n",
      "\"I know, I know,\" you say. \"But you're not going to like this one. It's not a good movie. I mean, it's a really bad movie, you know? And I'm not sure if you've ever seen it before, so I'll just tell you what I think\n",
      "\n",
      "2 I don't know about you, but there's only one thing I want to do after a long day of work, and that's to sit down and watch a movie.\"\n",
      "\n",
      "\"I know, I know,\" you say. \"But you're not going to like this one. It's not a good movie. I mean, it's a really bad movie, you know? And I'm not sure if you've ever seen it before, so I'll just tell you what I thought\n",
      "\n",
      "3 I don't know about you, but there's only one thing I want to do after a long day of work, and that's to sit down and watch a movie.\"\n",
      "\n",
      "\"I know, I know,\" you say. \"But you're not going to like this one. It's not a good movie. I mean, it's a really bad movie, you know? And I'm not sure if you've ever seen it before, so I'll just tell you what I saw\n",
      "\n",
      "4 I don't know about you, but there's only one thing I want to do after a long day of work, and that's to sit down and watch a movie.\"\n",
      "\n",
      "\"I know, I know,\" you say. \"But you're not going to like this one. It's not a good movie. I mean, it's a really bad movie, you know? And I'm not sure if you'll like it, either. But I'll tell you this: I don\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Setting Beam Parameters for Large Model\n",
    "Large_Beam_Output = Large_Model.generate( Large_Model_Input, max_length = Maximum_Length, num_beams = 5, no_repeat_ngram_size = 2, num_return_sequences = 5, early_stopping = True )\n",
    "# Printing Output For Large Model\n",
    "for Index_Value, Instance_Beam_Output in enumerate(Large_Beam_Output):\n",
    "      print(Index_Value, Large_Tokenizer.decode(Instance_Beam_Output, skip_special_tokens=True))\n",
    "      print()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "a10b2cee",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0 I don't know about you, but there's only one thing I want to do after a long day of work, and that's go to the gym.\n",
      "\n",
      "I've been working out since I was a little kid, so I know what it's like to get in shape. But I've never been able to find a gym where I feel like I'm getting the most out of my time there. So I decided to start my own gym, which is why I started this blog.\n",
      "\n",
      "1 I don't know about you, but there's only one thing I want to do after a long day of work, and that's go to the gym.\n",
      "\n",
      "I've been working out since I was a little kid, so I know what it's like to get in shape. But I've never been able to find a gym where I feel like I'm getting the most out of my time there. So I decided to start my own gym in my hometown of Chicago, Illinois. I\n",
      "\n",
      "2 I don't know about you, but there's only one thing I want to do after a long day of work, and that's go to the gym.\n",
      "\n",
      "I've been working out since I was a little kid, so I know what it's like to get in shape. But I've never been able to find a gym where I feel like I'm getting the most out of my time there. So I decided to start my own gym in my hometown of Chicago, Illinois, called\n",
      "\n",
      "3 I don't know about you, but there's only one thing I want to do after a long day of work, and that's go to the gym.\n",
      "\n",
      "I've been working out since I was a little kid, so I know what it's like to get in shape. But I've never been able to find a gym where I feel like I'm getting the most out of my time there. So I decided to start my own gym in my hometown of Chicago, Illinois. It\n",
      "\n",
      "4 I don't know about you, but there's only one thing I want to do after a long day of work, and that's go to the gym.\n",
      "\n",
      "I've been working out since I was a little kid, so I know what it's like to get in shape. But I've never been able to find a gym where I feel like I'm getting the most out of my time there. So I decided to start my own gym in my hometown of Chicago, Illinois. And\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Setting Beam Parameters for Medium Model\n",
    "Medium_Beam_Output = Medium_Model.generate( Medium_Model_Input, max_length = Maximum_Length, num_beams = 5, no_repeat_ngram_size = 2, num_return_sequences = 5, early_stopping = True )\n",
    "# Printing Output For Medium Model\n",
    "for Index_Value, Instance_Beam_Output in enumerate(Medium_Beam_Output):\n",
    "      print(Index_Value, Medium_Tokenizer.decode(Instance_Beam_Output, skip_special_tokens=True))\n",
    "      print()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "f781c0ce",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0 I don't know about you, but there's only one thing I want to do after a long day of work, and that's to get out of bed.\"\n",
      "\n",
      "\"I'm not going to lie to you,\" she said. \"It's not like I've ever done anything like this before. It's just a matter of time before I get back to work.\"\n",
      "\n",
      "1 I don't know about you, but there's only one thing I want to do after a long day of work, and that's to get out of bed.\"\n",
      "\n",
      "\"I'm not going to lie to you,\" she said. \"It's not like I've ever done anything like this before. It's just a matter of time before I get back to work. I just want you to know that I love you.\"\n",
      "\n",
      "2 I don't know about you, but there's only one thing I want to do after a long day of work, and that's to get out of bed.\"\n",
      "\n",
      "\"I'm not going to lie to you,\" she said. \"It's not like I've ever done anything like this before. It's just a matter of time before I get back to work. I just want you to know that I'm here for you and I love you.\"\n",
      "\n",
      "3 I don't know about you, but there's only one thing I want to do after a long day of work, and that's to get out of bed.\"\n",
      "\n",
      "\"I'm not going to lie to you,\" she said. \"It's not like I've ever done anything like this before. It's just a matter of time before I get back to work. I just want you to know that I'm here to help you out.\"\n",
      "\n",
      "4 I don't know about you, but there's only one thing I want to do after a long day of work, and that's to get out of bed.\"\n",
      "\n",
      "\"I'm not going to lie to you,\" she said. \"It's not like I've ever done anything like this before. It's just a matter of time before I get back to work. I just want you to know that I'm here for you and I love you for it.\"\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Setting Beam Parameters for Small Model\n",
    "Small_Beam_Output = Small_Model.generate( Small_Model_Input, max_length = Maximum_Length, num_beams = 5, no_repeat_ngram_size = 2, num_return_sequences = 5, early_stopping = True )\n",
    "# Printing Output For Small Model\n",
    "for Index_Value, Instance_Beam_Output in enumerate(Small_Beam_Output):\n",
    "      print(Index_Value, Small_Tokenizer.decode(Instance_Beam_Output, skip_special_tokens=True))\n",
    "      print()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "30258ab4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0 I don't know about you, but there's only one thing I want to do after a long day of work, and that's to sit down and watch a movie.\"\n",
      "\n",
      "\"I know, I know,\" you say. \"But you're not going to like this one. It's not a good movie. I mean, it's a really bad movie, you know? And I'm not sure if you've ever seen it before, so I'll just tell you what I think it is. You see, this movie is about a guy who's trying to get his girlfriend back. And he has to go through a lot of trials and tribulations in order to accomplish his goal. But the thing is, he doesn't even know what he wants. He just wants to be with her. That's all he knows. So he goes through all of these trials to try and get her back, all the way to the end, where he finds out that she's dead.\n",
      "\n",
      "1 I don't know about you, but there's only one thing I want to do after a long day of work, and that's to sit down and watch a movie.\"\n",
      "\n",
      "\"I know, I know,\" you say. \"But you're not going to like this one. It's not a good movie. I mean, it's a really bad movie, you know? And I'm not sure if you've ever seen it before, so I'll just tell you what I think it is. You see, this movie is about a guy who's trying to get his girlfriend back. And he has to go through a lot of trials and tribulations in order to accomplish his goal. But the thing is, he doesn't even know what he wants. He just wants to be with her. That's all he knows. So he goes through all of these trials to try and get her back, all the way to the end, where he finally finds out what she wants,\n",
      "\n",
      "2 I don't know about you, but there's only one thing I want to do after a long day of work, and that's to sit down and watch a movie.\"\n",
      "\n",
      "\"I know, I know,\" you say. \"But you're not going to like this one. It's not a good movie. I mean, it's a really bad movie, you know? And I'm not sure if you've ever seen it before, so I'll just tell you what I think it is. You see, this movie is about a guy who's trying to get his girlfriend back. And he has to go through a lot of trials and tribulations in order to accomplish his goal. But the thing is, he doesn't even know what he wants. He just wants to be with her. That's all he knows. So he goes through all of these trials to try and get her back, all the way to the end, where he finally finds out what she wants.\"\n",
      "\n",
      "3 I don't know about you, but there's only one thing I want to do after a long day of work, and that's to sit down and watch a movie.\"\n",
      "\n",
      "\"I know, I know,\" you say. \"But you're not going to like this one. It's not a good movie. I mean, it's a really bad movie, you know? And I'm not sure if you've ever seen it before, so I'll just tell you what I think it is. You see, this movie is about a guy who's trying to get his girlfriend back. And he has to go through a lot of trials and tribulations in order to accomplish his goal. But the thing is, he doesn't even know what he wants. He just wants to be with her. That's all he knows. So he goes through all of these trials to try and get her back, all the way to the end, where he finally finds out what she wants and\n",
      "\n",
      "4 I don't know about you, but there's only one thing I want to do after a long day of work, and that's to sit down and watch a movie.\"\n",
      "\n",
      "\"I know, I know,\" you say. \"But you're not going to like this one. It's not a good movie. I mean, it's a really bad movie, you know? And I'm not sure if you've ever seen it before, so I'll just tell you what I think it is. You see, this movie is about a guy who's trying to get his girlfriend back. And he has to go through a lot of trials and tribulations in order to accomplish his goal. But the thing is, he doesn't even know what he wants. He just wants to be with her. That's all he knows. So he goes through all of these trials to try and get her back, all the way to the end, where he finds out that she's dead.\"\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Try changing the Maximum Length of text Generation\n",
    "Maximum_Length = 200\n",
    "\n",
    "# Setting Beam Parameters for Large Model\n",
    "Large_Beam_Output = Large_Model.generate( Large_Model_Input, max_length = Maximum_Length, num_beams = 5, no_repeat_ngram_size = 2, num_return_sequences = 5, early_stopping = True )\n",
    "# Printing Output For Large Model\n",
    "for Index_Value, Instance_Beam_Output in enumerate(Large_Beam_Output):\n",
    "      print(Index_Value, Large_Tokenizer.decode(Instance_Beam_Output, skip_special_tokens=True))\n",
    "      print()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "9a93e140",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0 I don't know about you, but there's only one thing I want to do after a long day of work, and that's go to the gym.\n",
      "\n",
      "I've been working out since I was a little kid, so I know what it's like to get in shape. But I've never been able to find a gym where I feel like I'm getting the most out of my time there. So I decided to start my own gym in my hometown of Chicago, Illinois. I wanted to create a place where people could come in and get a good workout without feeling like they're wasting their time. That's why I started my gym, because I felt like it would be a great way for people to connect with each other and have fun while doing something they love.\n",
      "\n",
      "1 I don't know about you, but there's only one thing I want to do after a long day of work, and that's go to the gym.\n",
      "\n",
      "I've been working out since I was a little kid, so I know what it's like to get in shape. But I've never been able to find a gym where I feel like I'm getting the most out of my time there. So I decided to start my own gym in my hometown of Chicago, Illinois. I wanted to create a place where people could come in and get a good workout without feeling like they're wasting their time. That's why I started my gym, because I felt like it would be a great way for people to connect with each other and make new friends. It would also give me the opportunity to work out with some of the best athletes in the world, which is something I never thought I could do.\n",
      "\n",
      "2 I don't know about you, but there's only one thing I want to do after a long day of work, and that's go to the gym.\n",
      "\n",
      "I've been working out since I was a little kid, so I know what it's like to get in shape. But I've never been able to find a gym where I feel like I'm getting the most out of my time there. So I decided to start my own gym in my hometown of Chicago, Illinois. I wanted to create a place where people could come in and get a good workout without feeling like they're wasting their time. That's why I started my gym, because I felt like it would be a great way for people to connect with each other and make new friends. It would also give me the opportunity to work out with some of the best athletes in the world, which is something I never thought I'd do.\n",
      "\n",
      "3 I don't know about you, but there's only one thing I want to do after a long day of work, and that's go to the gym.\n",
      "\n",
      "I've been working out since I was a little kid, so I know what it's like to get in shape. But I've never been able to find a gym where I feel like I'm getting the most out of my time there. So I decided to start my own gym in my hometown of Chicago, Illinois. I wanted to create a place where people could come in and get a good workout without feeling like they're wasting their time. That's why I started my gym, because I felt like it would be a great way for people to connect with each other and make new friends. It would also give me the opportunity to work out with some of the best athletes in the world, which is something I never thought I'd get to experience.\n",
      "\n",
      "4 I don't know about you, but there's only one thing I want to do after a long day of work, and that's go to the gym.\n",
      "\n",
      "I've been working out since I was a little kid, so I know what it's like to get in shape. But I've never been able to find a gym where I feel like I'm getting the most out of my time there. So I decided to start my own gym in my hometown of Chicago, Illinois. I wanted to create a place where people could come in and get a good workout without feeling like they're wasting their time. That's why I started my gym, because I felt like it would be a great way for people to connect with each other and make new friends. It would also give me the opportunity to work out with some of the best athletes in the world, which is something I never thought I would get to experience.\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Setting Beam Parameters for Medium Model\n",
    "Medium_Beam_Output = Medium_Model.generate( Medium_Model_Input, max_length = Maximum_Length, num_beams = 5, no_repeat_ngram_size = 2, num_return_sequences = 5, early_stopping = True )\n",
    "# Printing Output For Medium Model\n",
    "for Index_Value, Instance_Beam_Output in enumerate(Medium_Beam_Output):\n",
    "      print(Index_Value, Medium_Tokenizer.decode(Instance_Beam_Output, skip_special_tokens=True))\n",
    "      print()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "731c8d1d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0 I don't know about you, but there's only one thing I want to do after a long day of work, and that's to get out of bed.\"\n",
      "\n",
      "\"I'm not going to lie to you,\" she said. \"It's not like I've ever done anything like this before. It's just a matter of time before I get back to work.\"\n",
      "\n",
      "1 I don't know about you, but there's only one thing I want to do after a long day of work, and that's to get out of bed.\"\n",
      "\n",
      "\"I'm not going to lie to you,\" she said. \"It's not like I've ever done anything like this before. It's just a matter of time before I get back to work. I just want you to know that I love you.\"\n",
      "\n",
      "2 I don't know about you, but there's only one thing I want to do after a long day of work, and that's to get out of bed.\"\n",
      "\n",
      "\"I'm not going to lie to you,\" she said. \"It's not like I've ever done anything like this before. It's just a matter of time before I get back to work. I just want you to know that I'm here for you and I love you.\"\n",
      "\n",
      "3 I don't know about you, but there's only one thing I want to do after a long day of work, and that's to get out of bed.\"\n",
      "\n",
      "\"I'm not going to lie to you,\" she said. \"It's not like I've ever done anything like this before. It's just a matter of time before I get back to work. I just want you to know that I'm here to help you out.\"\n",
      "\n",
      "4 I don't know about you, but there's only one thing I want to do after a long day of work, and that's to get out of bed.\"\n",
      "\n",
      "\"I'm not going to lie to you,\" she said. \"It's not like I've ever done anything like this before. It's just a matter of time before I get back to work. I just want you to know that I'm here for you and I love you for it.\"\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Setting Beam Parameters for Small Model\n",
    "Small_Beam_Output = Small_Model.generate( Small_Model_Input, max_length = Maximum_Length, num_beams = 5, no_repeat_ngram_size = 2, num_return_sequences = 5, early_stopping = True )\n",
    "# Printing Output For Small Model\n",
    "for Index_Value, Instance_Beam_Output in enumerate(Small_Beam_Output):\n",
    "      print(Index_Value, Small_Tokenizer.decode(Instance_Beam_Output, skip_special_tokens=True))\n",
    "      print()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "74f88fbc",
   "metadata": {},
   "source": [
    "# Trying Top K Sampling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "b1a78cae",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Setting the Maximum Length of text Generation and Temperature Value\n",
    "Maximum_Length = 100\n",
    "Temperature_Value = 0.8"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "14d543df",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Output of Large Model\n",
      "I don't know about you, but there's only one thing I want to do after a long day of work: put on a pair of lightweight jeans. (I don't know if I even have enough jeans on to wear them for another 30 days, though.) Lucky for me, I had a pair of these in my closet along with my Adi Dassler jeans, and I knew I needed to wear these if I was going to make it to the end of the week.\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Finding Sample Output for Large Model\n",
    "Top_K_Model_Large_Smaple_Output = Large_Model.generate( Large_Model_Input, do_sample = True, max_length = Maximum_Length, top_k = 0, temperature = Temperature_Value )\n",
    "print(\"Output of Large Model\")\n",
    "Top_K_Model_Large_Output = Large_Tokenizer.decode(Top_K_Model_Large_Smaple_Output[0], skip_special_tokens = True)\n",
    "print(Top_K_Model_Large_Output)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "825de9d3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Output of Medium Model\n",
      "I don't know about you, but there's only one thing I want to do after a long day of work. I want to drink a bottle of wine.\n",
      "\n",
      "I'm sure it's not the weird kind, but I started doing it last year. I'm not sure exactly when I started doing it, but I started doing it in May. I'm not sure of what bottle I started doing it with, but I think it was a cabernet. I'm trying to figure\n"
     ]
    }
   ],
   "source": [
    "# Finding Sample Output for Medium Model\n",
    "Top_K_Model_Medium_Smaple_Output = Medium_Model.generate( Medium_Model_Input, do_sample = True, max_length = Maximum_Length, top_k = 0, temperature = Temperature_Value )\n",
    "print(\"Output of Medium Model\")\n",
    "Top_K_Model_Medium_Output = Medium_Tokenizer.decode(Top_K_Model_Medium_Smaple_Output[0], skip_special_tokens = True)\n",
    "print(Top_K_Model_Medium_Output)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "eb5a7ac5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Output of Small Model\n",
      "I don't know about you, but there's only one thing I want to do after a long day of work: change my workout schedule. Besides of course adding another day of cooling off up, I love getting to work on this next Sunday. I actually learned this a couple days ago and am looking forward to continuing my training.\n",
      "\n",
      "One by one, I thank all of you who are continuing to participate in the transformation that I've been doing since last June. I believe you will also\n"
     ]
    }
   ],
   "source": [
    "# Finding Sample Output for Small Model\n",
    "Top_K_Model_Small_Smaple_Output = Small_Model.generate( Small_Model_Input, do_sample = True, max_length = Maximum_Length, top_k = 0, temperature = Temperature_Value )\n",
    "print(\"Output of Small Model\")\n",
    "Top_K_Model_Small_Output = Small_Tokenizer.decode(Top_K_Model_Small_Smaple_Output[0], skip_special_tokens = True)\n",
    "print(Top_K_Model_Small_Output)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "97b8ef86",
   "metadata": {},
   "outputs": [],
   "source": [
    "# changing temperature and Maximum Length \n",
    "Maximum_Length = 200\n",
    "Temperature_Value = 0.5"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "28f116de",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Output of Large Model\n",
      "I don't know about you, but there's only one thing I want to do after a long day of work: I want to get a little sleep. And I want to do it before I start working on the next thing I want to do.\n",
      "\n",
      "So I decided to get a good night's sleep before I start my next project.\n",
      "\n",
      "I've never been a fan of the \"get a good night's sleep before you start work\" method.\n",
      "\n",
      "I've never been a fan of the \"get a good night's sleep before you start work\" method.\n",
      "\n",
      "I've never been a fan of the \"get a good night's sleep before you start work\" method.\n",
      "\n",
      "I've never been a fan of the \"get a good night's sleep before you start work\" method.\n",
      "\n",
      "I've never been a fan of the \"get a good night's sleep before you start work\" method.\n",
      "\n",
      "I've never been a fan of the\n"
     ]
    }
   ],
   "source": [
    "# Finding Sample Output for Large Model\n",
    "Top_K_Model_Large_Smaple_Output = Large_Model.generate( Large_Model_Input, do_sample = True, max_length = Maximum_Length, top_k = 0, temperature = Temperature_Value )\n",
    "print(\"Output of Large Model\")\n",
    "Top_K_Model_Large_Output = Large_Tokenizer.decode(Top_K_Model_Large_Smaple_Output[0], skip_special_tokens = True)\n",
    "print(Top_K_Model_Large_Output)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "8f834e0f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Output of Medium Model\n",
      "I don't know about you, but there's only one thing I want to do after a long day of work.\"\n",
      "\n",
      "\"Yeah, well, I'm going to take the train home.\"\n",
      "\n",
      "\"I'm going to have to go to the gym. I'm going to have to get my body ready for the gym.\"\n",
      "\n",
      "\"I'll have to go to the gym, too.\"\n",
      "\n",
      "\"I'll have to go to the gym.\"\n",
      "\n",
      "\"I'll have to go to the gym.\"\n",
      "\n",
      "\"I'll have to go to the gym.\"\n",
      "\n",
      "\"I'll have to go to the gym.\"\n",
      "\n",
      "\"I'll have to go to the gym.\"\n",
      "\n",
      "\"I'll have to go to the gym.\"\n",
      "\n",
      "\"I'll have to go to the gym.\"\n",
      "\n",
      "\"I'll have to go to the gym.\"\n",
      "\n",
      "\"I'll have to go to the gym.\"\n",
      "\n",
      "\"I'll have to go to the\n"
     ]
    }
   ],
   "source": [
    "# Finding Sample Output for Medium Model\n",
    "Top_K_Model_Medium_Smaple_Output = Medium_Model.generate( Medium_Model_Input, do_sample = True, max_length = Maximum_Length, top_k = 0, temperature = Temperature_Value )\n",
    "print(\"Output of Medium Model\")\n",
    "Top_K_Model_Medium_Output = Medium_Tokenizer.decode(Top_K_Model_Medium_Smaple_Output[0], skip_special_tokens = True)\n",
    "print(Top_K_Model_Medium_Output)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "8b79728f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Output of Small Model\n",
      "I don't know about you, but there's only one thing I want to do after a long day of work. I want to get out into the field. I want to get out in the field with the right mindset and I want to go out and play. I want to get out there and give my all. I want to play like they say: 'He's going to get you out.'\"\n",
      "\n",
      "It's a message that has resonated with the players involved in the trade of the first round pick.\n",
      "\n",
      "\"It's a message that I've heard from a lot of players,\" said Hester. \"It's a message that I've heard from a lot of players. It's a message that I've heard from a lot of coaches. It's a message that I've heard from a lot of coaches, and I've learned a lot from them. They're just like, 'You're going to be a big part of this, so we're going to\n"
     ]
    }
   ],
   "source": [
    "# Finding Sample Output for Small Model\n",
    "Top_K_Model_Small_Smaple_Output = Small_Model.generate( Small_Model_Input, do_sample = True, max_length = Maximum_Length, top_k = 0, temperature = Temperature_Value )\n",
    "print(\"Output of Small Model\")\n",
    "Top_K_Model_Small_Output = Small_Tokenizer.decode(Top_K_Model_Small_Smaple_Output[0], skip_special_tokens = True)\n",
    "print(Top_K_Model_Small_Output)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b6a967a5",
   "metadata": {},
   "source": [
    "# Trying Top P Sampling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "42fa25af",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Setting the Maximum Length of text Generation and Top P Value\n",
    "Maximum_Length = 100\n",
    "Top_P_Value = 0.8"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "e723f8c5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Output of Large Model\n",
      "I don't know about you, but there's only one thing I want to do after a long day of work and after a long vacation: I want to enjoy life. Well, if you can, it's a nice one.\n",
      "\n",
      "You know what's just fun? Being with my family. My husband and I love being around each other. We spend most of our time together being active and having a great time.\n",
      "\n",
      "Besides that, my life is a good one. I don\n"
     ]
    }
   ],
   "source": [
    "# Finding Sample Output for Large Model\n",
    "Top_P_Model_Large_Smaple_Output = Large_Model.generate( Large_Model_Input, do_sample = True, max_length = Maximum_Length, top_k = 0, top_p = Top_P_Value )\n",
    "print(\"Output of Large Model\")\n",
    "Top_P_Model_Large_Output = Large_Tokenizer.decode(Top_P_Model_Large_Smaple_Output[0], skip_special_tokens = True)\n",
    "print(Top_P_Model_Large_Output)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "383b9dd0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Output of Medium Model\n",
      "I don't know about you, but there's only one thing I want to do after a long day of work and drinking. I want to get some pussy, and I want it a lot more than your cock. You're only going to want my pussy now.\"\n",
      "\n",
      "\n",
      "\"But I didn't get a chance to fuck your dick before you made my day.\"\n",
      "\n",
      "\n",
      "\"Well, now I have. I'm about to start.\" I'm almost choked with delight, but the chag\n"
     ]
    }
   ],
   "source": [
    "# Finding Sample Output for Medium Model\n",
    "Top_P_Model_Medium_Smaple_Output = Medium_Model.generate( Medium_Model_Input, do_sample = True, max_length = Maximum_Length, top_k = 0, top_p = Top_P_Value )\n",
    "print(\"Output of Medium Model\")\n",
    "Top_P_Model_Medium_Output = Medium_Tokenizer.decode(Top_P_Model_Medium_Smaple_Output[0], skip_special_tokens = True)\n",
    "print(Top_P_Model_Medium_Output)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "f5c51b11",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Output of Small Model\n",
      "I don't know about you, but there's only one thing I want to do after a long day of work.\"\n",
      "\n",
      "Blake sighs as she begins to climb up the bed, taking off her clothes and placing them on the bed. Her feet are soft, hard and warm. Blake has never felt so wet before. The moonlight shoots through the sandpaper and a small amount of warm sweat has dripped from her cheeks, revealing her small, white eyes and nose. She holds out\n"
     ]
    }
   ],
   "source": [
    "# Finding Sample Output for Small Model\n",
    "Top_P_Model_Small_Smaple_Output = Small_Model.generate( Small_Model_Input, do_sample = True, max_length = Maximum_Length, top_k = 0, top_p = Top_P_Value )\n",
    "print(\"Output of Small Model\")\n",
    "Top_P_Model_Small_Output = Small_Tokenizer.decode(Top_P_Model_Small_Smaple_Output[0], skip_special_tokens = True)\n",
    "print(Top_P_Model_Small_Output)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "f0c4b653",
   "metadata": {},
   "outputs": [],
   "source": [
    "# changing Top P Value and Maximum Length \n",
    "Maximum_Length = 200\n",
    "Top_P_Value = 0.5"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "2baf136c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Output of Large Model\n",
      "I don't know about you, but there's only one thing I want to do after a long day of work: watch a movie. And I'm not talking about the standard, popcorn-y \"Star Wars\" or \"The Hunger Games\" fare. I'm talking about a real film that's worth your time.\n",
      "\n",
      "There's a new movie out right now called \"The Greatest Showman.\" It's the story of the greatest stuntman of all time, the man who made the famous film of \"The Wizard of Oz\" with his own hands.\n",
      "\n",
      "In the movie, he takes on the role of the Tin Man, a big, green character with a long beard and a big red bowler hat. He's always trying to do the impossible, but the only thing that can stop him is the Tin Man himself.\n",
      "\n",
      "But it's not just any Tin Man. It's the Tin Man who is in fact a fictional character created by Disney, the same company\n"
     ]
    }
   ],
   "source": [
    "# Finding Sample Output for Large Model\n",
    "Top_P_Model_Large_Smaple_Output = Large_Model.generate( Large_Model_Input, do_sample = True, max_length = Maximum_Length, top_k = 0, top_p = Top_P_Value )\n",
    "print(\"Output of Large Model\")\n",
    "Top_P_Model_Large_Output = Large_Tokenizer.decode(Top_P_Model_Large_Smaple_Output[0], skip_special_tokens = True)\n",
    "print(Top_P_Model_Large_Output)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "4a62f4e3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Output of Medium Model\n",
      "I don't know about you, but there's only one thing I want to do after a long day of work, and that's go home and relax. And that's exactly what I'm doing right now.\"\n",
      "\n",
      "\"That's the point,\" replied the rabbit, before he ran off into the forest.\n",
      "\n",
      "After a while, the rabbit came back, but he was a bit more agitated than usual. He looked at his owner with a hint of worry in his eyes.\n",
      "\n",
      "\"Are you okay?\" asked the rabbit, who was a bit calmer than usual.\n",
      "\n",
      "\"I'm fine, but it's still scary,\" said the rabbit, who was worried about the bunny.\n",
      "\n",
      "\"That's because I'm a rabbit,\" replied the rabbit, who was a bit nervous about the rabbit.\n",
      "\n",
      "\"A rabbit?\" asked the rabbit, who was not familiar with the rabbit species.\n",
      "\n",
      "\"Yeah, a rabbit,\" replied the rabbit, who was a\n"
     ]
    }
   ],
   "source": [
    "# Finding Sample Output for Medium Model\n",
    "Top_P_Model_Medium_Smaple_Output = Medium_Model.generate( Medium_Model_Input, do_sample = True, max_length = Maximum_Length, top_k = 0, top_p = Top_P_Value )\n",
    "print(\"Output of Medium Model\")\n",
    "Top_P_Model_Medium_Output = Medium_Tokenizer.decode(Top_P_Model_Medium_Smaple_Output[0], skip_special_tokens = True)\n",
    "print(Top_P_Model_Medium_Output)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "0c26258b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Output of Small Model\n",
      "I don't know about you, but there's only one thing I want to do after a long day of work: I want to go home. I want to go home and work. I want to get my hands dirty. I want to work on this idea of doing it with a lot of effort.\n",
      "\n",
      "\"But I'm not going to do it until I've done it. I'm not going to do it until I've got my feet wet. I'm not going to do it until I've got my hands wet. I'm not going to do it until I've got my feet wet. I'm not going to do it until I've got my hands wet. I'm not going to do it until I've got my feet wet. I'm not going to do it until I've got my feet wet. I'm not going to do it until I've got my feet wet. I'm not going to do it until I've got my feet wet. I\n"
     ]
    }
   ],
   "source": [
    "# Finding Sample Output for Small Model\n",
    "Top_P_Model_Small_Smaple_Output = Small_Model.generate( Small_Model_Input, do_sample = True, max_length = Maximum_Length, top_k = 0, top_p = Top_P_Value )\n",
    "print(\"Output of Small Model\")\n",
    "Top_P_Model_Small_Output = Small_Tokenizer.decode(Top_P_Model_Small_Smaple_Output[0], skip_special_tokens = True)\n",
    "print(Top_P_Model_Small_Output)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7edeab10",
   "metadata": {},
   "source": [
    "# Trying P and K Sampling Both"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "0afbd56d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# setting top k and top p values & Maximum Length, Count of stories we need also\n",
    "Instance_P_Value = 0.8\n",
    "Instance_K_Value = 50\n",
    "Maximum_Length = 200\n",
    "Story_Count = 3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "fae53c13",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Output of Large Model - - - - -\n",
      "0 I don't know about you, but there's only one thing I want to do after a long day of work: sit back and read a good book. I've already read most of them, and I can't wait to read some new ones. I think it will be good to have a few more new books in the mix.\n",
      "\n",
      "1 I don't know about you, but there's only one thing I want to do after a long day of work: play some games. It's why I play the game I love, and I'm sure you do too. In fact, there's an entire segment of our community, and the majority of our players, that are gamers. That's why we're here, and we can't wait to see what you'll be able to do with our awesome engine.\"\n",
      "\n",
      "With a year and a half of open beta under their belts, Star Citizen will likely continue to be one of the biggest gaming events of the year. There's still plenty of work to be done, and we'll be keeping our eyes open for more announcements from Cloud Imperium Games in the weeks ahead.\n",
      "\n",
      "Update: The video has been pulled from the YouTube page, but can still be found here.\n",
      "\n",
      "2 I don't know about you, but there's only one thing I want to do after a long day of work: grab a cold beer. That's right, I'm talking about the real deal. If you're in need of a good cold beer, this is the place to go.\n",
      "\n",
      "P.S. We don't sell liquor and the place is not licensed for them.\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Finding Sample Output for Large Model\n",
    "Top_PnK_Model_Large_Smaple_Output = Large_Model.generate( Large_Model_Input, do_sample = True, max_length = Maximum_Length, top_k = Instance_K_Value, top_p = Instance_P_Value, num_return_sequences = Story_Count )\n",
    "print(\"Output of Large Model - - - - -\")\n",
    "# Printing Output For Large Model\n",
    "for Index_Value, Instance_Output in enumerate(Top_PnK_Model_Large_Smaple_Output):\n",
    "      print(Index_Value, Large_Tokenizer.decode(Instance_Output, skip_special_tokens=True))\n",
    "      print()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "9bf346fc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Output of Medium Model - - - - -\n",
      "0 I don't know about you, but there's only one thing I want to do after a long day of work and I want to eat some food, right? Well, if you're my friend, then that's your problem. You don't need to ask for permission to eat food.\n",
      "\n",
      "And don't worry. I'm not going to tell you how to do that, or what to do with that food or what to do with all those leftover scraps of food you threw away. No, I'm going to teach you how to make your own meals from scratch using fresh ingredients you find on the streets and in the trash.\n",
      "\n",
      "Here's the catch, though: I'm not talking about a recipe or a fancy recipe here. I'm not talking about a recipe for a pizza that has been sitting in a tin for months and never made it to the store because the store owner thinks the ingredients are too expensive. I'm not even talking about a recipe that's just like\n",
      "\n",
      "1 I don't know about you, but there's only one thing I want to do after a long day of work: go to the bathroom.\n",
      "\n",
      "A few weeks ago, I was on the phone with a friend about something. I asked her if she had any more ideas, and she told me that she had one. I told her I had one too.\n",
      "\n",
      "A few minutes later, I got a call from her. She wanted to know if I had any ideas. She said she'd get back to me. She didn't want to talk about it. She wanted to get this over with.\n",
      "\n",
      "Then a week later, my friend called back. She had the same idea: she wanted to go to the bathroom and have a shower. But she'd get to that later.\n",
      "\n",
      "So I was at my friend's house this morning when I got the phone call. My heart was beating really fast. I knew she had a plan. But it wasn't one\n",
      "\n",
      "2 I don't know about you, but there's only one thing I want to do after a long day of work: take a shower. I'm not talking about a shower in the shower, just the shower in my office.\"\n",
      "\n",
      "I'll let you think about it for a bit.\n",
      "\n",
      "If that's how you'd feel about taking a shower after your commute, you're in luck: this is how most people feel after a long day of work.\n",
      "\n",
      "A recent study out of Boston University found that a shower with hot water can be \"positively associated\" with a more productive day. A recent study out of Boston University found that a shower with hot water can be \"positively associated\" with a more productive day.\n",
      "\n",
      "We don't like to think of ourselves as \"happiest people in the world,\" but most of us are happier when we can shower with hot water.\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Finding Sample Output for Medium Model\n",
    "Top_PnK_Model_Medium_Smaple_Output = Medium_Model.generate( Medium_Model_Input, do_sample = True, max_length = Maximum_Length, top_k = Instance_K_Value, top_p = Instance_P_Value, num_return_sequences = Story_Count )\n",
    "print(\"Output of Medium Model - - - - -\")\n",
    "# Printing Output For Medium Model\n",
    "for Index_Value, Instance_Output in enumerate(Top_PnK_Model_Medium_Smaple_Output):\n",
    "      print(Index_Value, Medium_Tokenizer.decode(Instance_Output, skip_special_tokens=True))\n",
    "      print()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "35ee990b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Output of Small Model - - - - -\n",
      "0 I don't know about you, but there's only one thing I want to do after a long day of work: Make sure my dad gets the rest of the week off.\n",
      "\n",
      "So I'm going to do a little bit of what I have to do to prepare for that day. I'm going to do what I've got to do to prepare for this weekend. That's what I'm going to do.\n",
      "\n",
      "What are you doing for this weekend?\n",
      "\n",
      "What are you doing for this weekend?\n",
      "\n",
      "Do you like this game?\n",
      "\n",
      "What are you doing for this weekend?\n",
      "\n",
      "What are you doing for this weekend?\n",
      "\n",
      "What are you doing for this weekend?\n",
      "\n",
      "Do you like this game?\n",
      "\n",
      "Do you like this game?\n",
      "\n",
      "What are you doing for this weekend?\n",
      "\n",
      "What are you doing for this weekend?\n",
      "\n",
      "What are you doing for this weekend?\n",
      "\n",
      "What are you doing for this weekend?\n",
      "\n",
      "\n",
      "1 I don't know about you, but there's only one thing I want to do after a long day of work: make a phone call. I'm very thankful for this opportunity.\"\n",
      "\n",
      "In February, Facebook announced it would take a year to update its mobile app with new features and add new features, including the ability to view and update photos and videos on the app. It's one step in a process that's already begun with Twitter and other social media platforms.\n",
      "\n",
      "\"It's really about time that Facebook took its time and put out a real, tangible product that people love,\" said Peter Pinchbeck, president and chief operating officer of Facebook Messenger. \"I'm really proud that Facebook was able to help people build their Facebook experience, and that it's the first mobile app to bring this kind of experience to iOS, and that we're going to continue to build and grow the mobile experience to reach more and more people. It's really been an amazing journey for our app\n",
      "\n",
      "2 I don't know about you, but there's only one thing I want to do after a long day of work. I want to be able to enjoy life without worrying about other people.\n",
      "\n",
      "\"If I were to make it, it would take years of hard work. The work will be hard, but my life will be full of meaning and fun. I can't imagine how my life would feel like without my work.\n",
      "\n",
      "\"My kids, their whole lives have been on my shoulders. When I look back at them now, I am happy and optimistic. I wish I could do more of this for them.\n",
      "\n",
      "\"I would like to see them become the people they were meant to be. I want to see them grow up and see them live their life with confidence. That's the way it should be.\n",
      "\n",
      "\"I love the way they look at me. It's all my life, but they've always been there for me. When I meet them\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Finding Sample Output for Small Model\n",
    "Top_PnK_Model_Small_Smaple_Output = Small_Model.generate( Small_Model_Input, do_sample = True, max_length = Maximum_Length, top_k = Instance_K_Value, top_p = Instance_P_Value, num_return_sequences = Story_Count )\n",
    "print(\"Output of Small Model - - - - -\")\n",
    "# Printing Output For Small Model\n",
    "for Index_Value, Instance_Output in enumerate(Top_PnK_Model_Small_Smaple_Output):\n",
    "      print(Index_Value, Small_Tokenizer.decode(Instance_Output, skip_special_tokens=True))\n",
    "      print()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "9d7725f0",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Changing top k and top p values & Maximum Length, Count of stories we need also\n",
    "Instance_P_Value = 0.5\n",
    "Instance_K_Value = 60\n",
    "Maximum_Length = 400\n",
    "Story_Count = 6"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "af5846ac",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Output of Large Model - - - - -\n",
      "0 I don't know about you, but there's only one thing I want to do after a long day of work. I want to sit down and watch a movie. I don't want to get up and do anything.\n",
      "\n",
      "So I've decided to go to the movies and enjoy myself. And I'm going to get a good night's sleep.\n",
      "\n",
      "And I'm going to get a nice, long, hot shower.\n",
      "\n",
      "And I'm going to take a nap.\n",
      "\n",
      "And I'm going to have a nice, long, hot shower.\n",
      "\n",
      "And I'm going to get up and do it all over again.\n",
      "\n",
      "Because I want to.\n",
      "\n",
      "I want to.\n",
      "\n",
      "I want to.\n",
      "\n",
      "I want to.\n",
      "\n",
      "I want to.\n",
      "\n",
      "I want to.\n",
      "\n",
      "I want to.\n",
      "\n",
      "I want to.\n",
      "\n",
      "I want to.\n",
      "\n",
      "I want to.\n",
      "\n",
      "I want to.\n",
      "\n",
      "I want to.\n",
      "\n",
      "I want to.\n",
      "\n",
      "I want to.\n",
      "\n",
      "I want to.\n",
      "\n",
      "I want to.\n",
      "\n",
      "I want to.\n",
      "\n",
      "I want to.\n",
      "\n",
      "I want to.\n",
      "\n",
      "I want to.\n",
      "\n",
      "I want to.\n",
      "\n",
      "I want to.\n",
      "\n",
      "I want to.\n",
      "\n",
      "I want to.\n",
      "\n",
      "I want to.\n",
      "\n",
      "I want to.\n",
      "\n",
      "I want to.\n",
      "\n",
      "I want to.\n",
      "\n",
      "I want to.\n",
      "\n",
      "I want to.\n",
      "\n",
      "I want to.\n",
      "\n",
      "I want to.\n",
      "\n",
      "I want to.\n",
      "\n",
      "I want to.\n",
      "\n",
      "I want to.\n",
      "\n",
      "I want to.\n",
      "\n",
      "I want to.\n",
      "\n",
      "I want to.\n",
      "\n",
      "I want to.\n",
      "\n",
      "I want to.\n",
      "\n",
      "I want to.\n",
      "\n",
      "I want to.\n",
      "\n",
      "I want to.\n",
      "\n",
      "I want to.\n",
      "\n",
      "I want\n",
      "\n",
      "1 I don't know about you, but there's only one thing I want to do after a long day of work: go to the gym. I've been working out since I was a kid, but I never got the chance to get in a proper workout. That's why I'm excited to finally get my workout in.\n",
      "\n",
      "Advertisement\n",
      "\n",
      "I'm going to do some work on my form today, but I'm going to do some work on my technique as well. I'm going to do a couple of sets of 15-20 reps, and I'm going to do a couple of sets of 10-15 reps. I'm going to do a couple of sets of 10-15 reps, and I'm going to do a couple of sets of 5-8 reps. I'm going to do a couple of sets of 5-8 reps, and I'm going to do a couple of sets of 3-5 reps. I'm going to do a couple of sets of 3-5 reps, and I'm going to do a couple of sets of 1-2 reps. I'm going to do a couple of sets of 1-2 reps, and I'm going to do a couple of sets of 5-8 reps. I'm going to do a couple of sets of 5-8 reps, and I'm going to do a couple of sets of 3-5 reps. I'm going to do a couple of sets of 3-5 reps, and I'm going to do a couple of sets of 1-2 reps. I'm going to do a couple of sets of 1-2 reps, and I'm going to do a couple of sets of 5-8 reps. I'm going to do a couple of sets of 5-8 reps, and I'm going to do a couple of sets of 3-5 reps. I'm going to do a couple of sets of 3-5 reps, and I'm going to do a couple of\n",
      "\n",
      "2 I don't know about you, but there's only one thing I want to do after a long day of work: watch a movie.\"\n",
      "\n",
      "\"I'm sure it's great, but I don't know about you, but I don't want to watch it.\"\n",
      "\n",
      "\"I don't know about you, but I don't want to do it.\"\n",
      "\n",
      "\"I don't know about you, but I don't want to do it.\"\n",
      "\n",
      "\"I don't know about you, but I don't want to do it.\"\n",
      "\n",
      "\"I don't know about you, but I don't want to do it.\"\n",
      "\n",
      "\"I don't know about you, but I don't want to do it.\"\n",
      "\n",
      "\"I don't know about you, but I don't want to do it.\"\n",
      "\n",
      "\"I don't know about you, but I don't want to do it.\"\n",
      "\n",
      "\"I don't know about you, but I don't want to do it.\"\n",
      "\n",
      "\"I don't know about you, but I don't want to do it.\"\n",
      "\n",
      "\"I don't know about you, but I don't want to do it.\"\n",
      "\n",
      "\"I don't know about you, but I don't want to do it.\"\n",
      "\n",
      "\"I don't know about you, but I don't want to do it.\"\n",
      "\n",
      "\"I don't know about you, but I don't want to do it.\"\n",
      "\n",
      "\"I don't know about you, but I don't want to do it.\"\n",
      "\n",
      "\"I don't know about you, but I don't want to do it.\"\n",
      "\n",
      "\"I don't know about you, but I don't want to do it.\"\n",
      "\n",
      "\"I don't know about you, but I don't want to do it.\"\n",
      "\n",
      "\"I don't know about you, but I don't want to do it.\"\n",
      "\n",
      "\"\n",
      "\n",
      "3 I don't know about you, but there's only one thing I want to do after a long day of work: take a nice long shower.\n",
      "\n",
      "I've been doing this for a while now, and I've gotten really good at it. I don't have to scrub my face or shave my legs or anything like that. I just put a little bit of water on my face and rub it in. I don't even have to rinse it off. I can do it every day.\n",
      "\n",
      "I have a shower head that's really big and I can put water on it and it's like, \"OK, I've got my shower head.\" It's so convenient. I can just put it on and it's all I need.\n",
      "\n",
      "What do you think makes a good shower?\n",
      "\n",
      "I think a good shower is really simple. It's just a little bit of water, a little bit of soap, a little bit of a lotion, and then a little bit of a shampoo. It's just like that.\n",
      "\n",
      "If you want to get really good at it, you can get a little bit of a scrub. It's not that complicated. You can do it in your kitchen. It's just a little bit of water, a little bit of soap, a little bit of a lotion, and then a little bit of a shampoo.\n",
      "\n",
      "I think a lot of people are intimidated by the idea of getting a good shower. But if you really want to do it, it's really easy.\n",
      "\n",
      "You can get a really good shower at a really cheap price.\n",
      "\n",
      "How do you get the most out of a shower?\n",
      "\n",
      "I think a lot of people get a really good shower because they don't know what they're doing. I think it's just about learning to relax and not worry about it too much.\n",
      "\n",
      "I think it's really important to just relax and just relax. You don't have\n",
      "\n",
      "4 I don't know about you, but there's only one thing I want to do after a long day of work. I want to get a cup of coffee and have a chat with a friend. And I want to do it in a friendly, relaxed way. I want to chat with a friend, and I want to have a chat with a friend.\n",
      "\n",
      "I want to talk to a friend about my day. I want to talk to a friend about my day.\n",
      "\n",
      "And that's what I'm going to do.\n",
      "\n",
      "I'm going to chat with a friend about my day. I'm going to talk to a friend about my day.\n",
      "\n",
      "It's going to be a friendly, relaxed chat.\n",
      "\n",
      "I'm going to have a chat with a friend about my day. I'm going to have a chat with a friend about my day.\n",
      "\n",
      "I'm going to have a chat with a friend about my day. I'm going to have a chat with a friend about my day.\n",
      "\n",
      "And that's what I'm going to do.\n",
      "\n",
      "I'm going to have a chat with a friend about my day. I'm going to have a chat with a friend about my day.\n",
      "\n",
      "And that's what I'm going to do.\n",
      "\n",
      "I'm going to have a chat with a friend about my day. I'm going to have a chat with a friend about my day.\n",
      "\n",
      "And that's what I'm going to do.\n",
      "\n",
      "I'm going to have a chat with a friend about my day. I'm going to have a chat with a friend about my day.\n",
      "\n",
      "And that's what I'm going to do.\n",
      "\n",
      "I'm going to have a chat with a friend about my day. I'm going to have a chat with a friend about my day.\n",
      "\n",
      "And that's what I'm going to do.\n",
      "\n",
      "I'm going to have a chat with a friend about\n",
      "\n",
      "5 I don't know about you, but there's only one thing I want to do after a long day of work. I want to get home and relax, and then I want to get some food and a drink and a movie.\"\n",
      "\n",
      "He also wants to be able to take a shower.\n",
      "\n",
      "\"I've been told that it's really hard to get a shower in the middle of the night,\" he said. \"I want to be able to get a shower in the middle of the night. I want to be able to take a shower. I want to be able to take a shower. I want to be able to take a shower. I want to be able to take a shower. I want to be able to take a shower. I want to be able to take a shower. I want to be able to take a shower. I want to be able to take a shower. I want to be able to take a shower. I want to be able to take a shower. I want to be able to take a shower. I want to be able to take a shower. I want to be able to take a shower. I want to be able to take a shower. I want to be able to take a shower. I want to be able to take a shower. I want to be able to take a shower. I want to be able to take a shower. I want to be able to take a shower. I want to be able to take a shower. I want to be able to take a shower. I want to be able to take a shower. I want to be able to take a shower. I want to be able to take a shower. I want to be able to take a shower. I want to be able to take a shower. I want to be able to take a shower. I want to be able to take a shower. I want to be able to take a shower. I want to be able to take\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Finding Sample Output for Large Model\n",
    "Top_PnK_Model_Large_Smaple_Output = Large_Model.generate( Large_Model_Input, do_sample = True, max_length = Maximum_Length, top_k = Instance_K_Value, top_p = Instance_P_Value, num_return_sequences = Story_Count )\n",
    "print(\"Output of Large Model - - - - -\")\n",
    "# Printing Output For Large Model\n",
    "for Index_Value, Instance_Output in enumerate(Top_PnK_Model_Large_Smaple_Output):\n",
    "      print(Index_Value, Large_Tokenizer.decode(Instance_Output, skip_special_tokens=True))\n",
    "      print()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "d9bbf86f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Output of Medium Model - - - - -\n",
      "0 I don't know about you, but there's only one thing I want to do after a long day of work. I want to sleep.\n",
      "\n",
      "I'm tired of being tired. I'm tired of being sleepy. I'm tired of being on my feet all day. I'm tired of waking up at 5:30 in the morning and being at work for three hours. I'm tired of having to walk to the bus stop to get to my car. I'm tired of waking up at 5:30 in the morning and being at work for three hours. I'm tired of having to walk to the bus stop to get to my car. I'm tired of waking up at 5:30 in the morning and being at work for three hours. I'm tired of having to walk to the bus stop to get to my car. I'm tired of waking up at 5:30 in the morning and being at work for three hours. I'm tired of having to walk to the bus stop to get to my car. I'm tired of waking up at 5:30 in the morning and being at work for three hours. I'm tired of having to walk to the bus stop to get to my car. I'm tired of waking up at 5:30 in the morning and being at work for three hours. I'm tired of having to walk to the bus stop to get to my car. I'm tired of waking up at 5:30 in the morning and being at work for three hours. I'm tired of having to walk to the bus stop to get to my car. I'm tired of waking up at 5:30 in the morning and being at work for three hours. I'm tired of having to walk to the bus stop to get to my car. I'm tired of waking up at 5:30 in the morning and being at work for three hours. I'm tired of having to walk to the bus stop to get to my car. I'm\n",
      "\n",
      "1 I don't know about you, but there's only one thing I want to do after a long day of work, and that's watch the latest episode of The Walking Dead.\"\n",
      "\n",
      "\"The Walking Dead\" is one of the best shows on television right now. The series is packed with characters, some of which are quite complex and complex characters. There are some characters who have the ability to do some amazing things, but they also have the ability to do some awful things. There are some characters who are just plain bad, and there are some characters who are just plain good.\n",
      "\n",
      "There are some characters who are just plain bad, and there are some characters who are just plain good.\n",
      "\n",
      "I don't want to say that I don't care about the characters. I care about the show. I care about the characters. I care about the show. I care about the show. I care about the show. I care about the show. I care about the show. I care about the show. I care about the show. I care about the show. I care about the show. I care about the show. I care about the show. I care about the show. I care about the show. I care about the show. I care about the show. I care about the show. I care about the show. I care about the show. I care about the show. I care about the show. I care about the show. I care about the show. I care about the show. I care about the show. I care about the show. I care about the show. I care about the show. I care about the show. I care about the show. I care about the show. I care about the show. I care about the show. I care about the show. I care about the show. I care about the show. I care about the show. I care about the show. I care about the show. I care\n",
      "\n",
      "2 I don't know about you, but there's only one thing I want to do after a long day of work. I want to get home to my wife and kids and watch a movie.\"\n",
      "\n",
      "That's when I heard the words \"The Man Who Killed Christmas\" come out of his mouth.\n",
      "\n",
      "I've been thinking about this for a while.\n",
      "\n",
      "I was in a car accident in my early 20s. I was on my way to work when I was hit by a car. I was unconscious for two days, but the doctor said I would be fine.\n",
      "\n",
      "I remember thinking to myself, \"I'm not going to be okay.\"\n",
      "\n",
      "I'm a strong person. I'm not going to be okay.\n",
      "\n",
      "I've been in recovery for almost 10 years now. I'm not a person who gets upset easily. I'm a person who's never been afraid to fight back. I'm not going to let anyone tell me that I can't fight back.\n",
      "\n",
      "But I'm also a person who has a family to take care of. My wife and kids are the ones who are going to take care of me.\n",
      "\n",
      "I was sitting in the car when the car slammed into me. I was in the hospital for two weeks, and I've been out of the hospital for about a year. I've been working in retail, and I've been in the workforce for 20 years. I've never been in a car accident.\n",
      "\n",
      "I've never been hit by a car. I've never been in a car accident.\n",
      "\n",
      "I've never been in a car accident.\n",
      "\n",
      "I was in the hospital for two weeks, and I've been out of the hospital for about a year. I've been working in retail, and I've been in the workforce for 20 years. I've never been in a car accident.\n",
      "\n",
      "But I've also been in recovery for about 10 years now. I'm not\n",
      "\n",
      "3 I don't know about you, but there's only one thing I want to do after a long day of work: sleep.\n",
      "\n",
      "And it's not because I'm tired. It's because I want to be alone.\n",
      "\n",
      "That's why I'm here.\n",
      "\n",
      "I'm here to help you find your own peace and happiness.\n",
      "\n",
      "I want to share with you the tools I've used to help me get there.\n",
      "\n",
      "I want to share with you the ways in which I've worked to improve myself and the people around me.\n",
      "\n",
      "I want to share with you the way I've been able to do this all by myself.\n",
      "\n",
      "I want to share with you the reasons why I'm here.\n",
      "\n",
      "And most of all, I want to share with you the tools that have helped me to find peace and happiness in my life.\n",
      "\n",
      "The tools I've used to help me find peace and happiness in my life\n",
      "\n",
      "I've been living my life as a Christian.\n",
      "\n",
      "I've been living my life as a Christian.\n",
      "\n",
      "I've been living my life as a Christian.\n",
      "\n",
      "I've been living my life as a Christian.\n",
      "\n",
      "I've been living my life as a Christian.\n",
      "\n",
      "I've been living my life as a Christian.\n",
      "\n",
      "I've been living my life as a Christian.\n",
      "\n",
      "I've been living my life as a Christian.\n",
      "\n",
      "I've been living my life as a Christian.\n",
      "\n",
      "I've been living my life as a Christian.\n",
      "\n",
      "I've been living my life as a Christian.\n",
      "\n",
      "I've been living my life as a Christian.\n",
      "\n",
      "I've been living my life as a Christian.\n",
      "\n",
      "I've been living my life as a Christian.\n",
      "\n",
      "I've been living my life as a Christian.\n",
      "\n",
      "I've been living my life as a Christian.\n",
      "\n",
      "I've been living my life as a Christian.\n",
      "\n",
      "\n",
      "\n",
      "4 I don't know about you, but there's only one thing I want to do after a long day of work. I want to get out of the house and walk around the block, but I'm afraid to do that.\n",
      "\n",
      "I'm afraid to be alone.\n",
      "\n",
      "I'm afraid to be alone.\n",
      "\n",
      "I'm afraid to be alone.\n",
      "\n",
      "I'm afraid to be alone.\n",
      "\n",
      "I'm afraid to be alone.\n",
      "\n",
      "I'm afraid to be alone.\n",
      "\n",
      "I'm afraid to be alone.\n",
      "\n",
      "I'm afraid to be alone.\n",
      "\n",
      "I'm afraid to be alone.\n",
      "\n",
      "I'm afraid to be alone.\n",
      "\n",
      "I'm afraid to be alone.\n",
      "\n",
      "I'm afraid to be alone.\n",
      "\n",
      "I'm afraid to be alone.\n",
      "\n",
      "I'm afraid to be alone.\n",
      "\n",
      "I'm afraid to be alone.\n",
      "\n",
      "I'm afraid to be alone.\n",
      "\n",
      "I'm afraid to be alone.\n",
      "\n",
      "I'm afraid to be alone.\n",
      "\n",
      "I'm afraid to be alone.\n",
      "\n",
      "I'm afraid to be alone.\n",
      "\n",
      "I'm afraid to be alone.\n",
      "\n",
      "I'm afraid to be alone.\n",
      "\n",
      "I'm afraid to be alone.\n",
      "\n",
      "I'm afraid to be alone.\n",
      "\n",
      "I'm afraid to be alone.\n",
      "\n",
      "I'm afraid to be alone.\n",
      "\n",
      "I'm afraid to be alone.\n",
      "\n",
      "I'm afraid to be alone.\n",
      "\n",
      "I'm afraid to be alone.\n",
      "\n",
      "I'm afraid to be alone.\n",
      "\n",
      "I'm afraid to be alone.\n",
      "\n",
      "I'm afraid to be alone.\n",
      "\n",
      "I'm afraid to be alone.\n",
      "\n",
      "I'm afraid to be alone.\n",
      "\n",
      "I'm afraid to be alone.\n",
      "\n",
      "I'm afraid to be alone.\n",
      "\n",
      "I'm afraid to be alone.\n",
      "\n",
      "I'm afraid to be alone.\n",
      "\n",
      "I'm afraid to be alone.\n",
      "\n",
      "I\n",
      "\n",
      "5 I don't know about you, but there's only one thing I want to do after a long day of work, and that's drink some water.\"\n",
      "\n",
      "\"I'll be fine,\" I said. \"I've been drinking.\"\n",
      "\n",
      "\"You'll be fine, then,\" he said. \"You'll just have to get some rest. You'll have a good day.\"\n",
      "\n",
      "I nodded, then said, \"I'll see you tomorrow.\"\n",
      "\n",
      "He laughed. \"Good luck, then.\"\n",
      "\n",
      "\"I'll see you then,\" I said. \"See you tomorrow.\"\n",
      "\n",
      "I walked away.\n",
      "\n",
      "The next morning, I had a headache, but I wasn't in a bad mood.\n",
      "\n",
      "I had been in the hospital for two days.\n",
      "\n",
      "I got up and went to the front desk. I told the nurse that I had a headache, and that I wanted to see the doctor.\n",
      "\n",
      "\"I'll call you,\" she said.\n",
      "\n",
      "I hung up the phone and went to the bathroom. I washed my face and neck, and then I went to the bathroom.\n",
      "\n",
      "I went to the bathroom and got out of the shower. I washed my face and neck and put my clothes on. I went to the bathroom, and I went to the front desk. I told the nurse that I had a headache, and that I wanted to see the doctor.\n",
      "\n",
      "\"I'll call you,\" she said.\n",
      "\n",
      "I hung up the phone and went to the bathroom. I washed my face and neck and put my clothes on. I went to the bathroom, and I went to the front desk. I told the nurse that I had a headache, and that I wanted to see the doctor.\n",
      "\n",
      "\"I'll call you,\" she said.\n",
      "\n",
      "I hung up the phone and went to the bathroom. I washed my face and neck and put my clothes on. I went to the bathroom, and\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Finding Sample Output for Medium Model\n",
    "Top_PnK_Model_Medium_Smaple_Output = Medium_Model.generate( Medium_Model_Input, do_sample = True, max_length = Maximum_Length, top_k = Instance_K_Value, top_p = Instance_P_Value, num_return_sequences = Story_Count )\n",
    "print(\"Output of Medium Model - - - - -\")\n",
    "# Printing Output For Medium Model\n",
    "for Index_Value, Instance_Output in enumerate(Top_PnK_Model_Medium_Smaple_Output):\n",
    "      print(Index_Value, Medium_Tokenizer.decode(Instance_Output, skip_special_tokens=True))\n",
    "      print()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "dc185a6c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Output of Small Model - - - - -\n",
      "0 I don't know about you, but there's only one thing I want to do after a long day of work. I want to get out of here and start making a living. I want to be a better person. I want to be a better person. I want to be a better person. I want to be a better person. I want to be a better person. I want to be a better person. I want to be a better person. I want to be a better person. I want to be a better person. I want to be a better person. I want to be a better person. I want to be a better person. I want to be a better person. I want to be a better person. I want to be a better person. I want to be a better person. I want to be a better person. I want to be a better person. I want to be a better person. I want to be a better person. I want to be a better person. I want to be a better person. I want to be a better person. I want to be a better person. I want to be a better person. I want to be a better person. I want to be a better person. I want to be a better person. I want to be a better person. I want to be a better person. I want to be a better person. I want to be a better person. I want to be a better person. I want to be a better person. I want to be a better person. I want to be a better person. I want to be a better person. I want to be a better person. I want to be a better person. I want to be a better person. I want to be a better person. I want to be a better person. I want to be a better person. I want to be a better person. I want to be a better person. I want to\n",
      "\n",
      "1 I don't know about you, but there's only one thing I want to do after a long day of work, and that's to get to work. I want to get to work, and I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to get to work. I want to\n",
      "\n",
      "2 I don't know about you, but there's only one thing I want to do after a long day of work: get up and do something.\"\n",
      "\n",
      "\"What are you doing?\" asked the girl, who seemed to be a bit upset at her own lack of response.\n",
      "\n",
      "\"I'm going to take a shower, I'm going to get a haircut,\" she said, and then, with a wave of her hand, she took off her shoes and ran out the door.\n",
      "\n",
      "A few minutes later, she returned to her apartment, where she had to go to the bathroom. She was dressed in a black dress and black pants, and her hair was still tied in a ponytail.\n",
      "\n",
      "\"What's going on?\" asked the girl, who seemed to be a bit upset at her own lack of response.\n",
      "\n",
      "\"I just woke up in the middle of the night and it was like I was going to die,\" she said. \"I don't know what happened to me, but I just want to be out of here.\"\n",
      "\n",
      "\"I'm sorry,\" said the girl, who was also wearing a black dress and black pants. \"I just want to go to bed.\"\n",
      "\n",
      "\"What do you mean?\" asked the girl, who was also wearing a black dress and black pants.\n",
      "\n",
      "\"I'm just trying to figure out what's going on,\" she said. \"I'm trying to figure out what's going on. I'm just trying to get back to work.\"\n",
      "\n",
      "\"What's going on?\" asked the girl, who was also wearing a black dress and black pants.\n",
      "\n",
      "\"I'm just trying to figure out what's going on,\" she said. \"I'm trying to figure out what's going on.\"\n",
      "\n",
      "\"What's going on?\" asked the girl, who was also wearing a black dress and black pants.\n",
      "\n",
      "\"I'm just trying to figure out what's going on,\"\n",
      "\n",
      "3 I don't know about you, but there's only one thing I want to do after a long day of work. I'm going to start work on a new book, and I'm going to write a new book. I don't know if I'm going to do it right now. I don't know if I'm going to write a book. I don't know if I'm going to do a book. I don't know if I'm going to write a book. I don't know if I'm going to write a book. I don't know if I'm going to write a book. I don't know if I'm going to write a book. I don't know if I'm going to write a book. I don't know if I'm going to write a book. I don't know if I'm going to write a book. I don't know if I'm going to write a book. I don't know if I'm going to write a book. I don't know if I'm going to write a book. I don't know if I'm going to write a book. I don't know if I'm going to write a book. I don't know if I'm going to write a book. I don't know if I'm going to write a book. I don't know if I'm going to write a book. I don't know if I'm going to write a book. I don't know if I'm going to write a book. I don't know if I'm going to write a book. I don't know if I'm going to write a book. I don't know if I'm going to write a book. I don't know if I'm going to write a book. I don't know if I'm going to write a book. I don't know if I'm going to write a book. I don't know if I'm going to write a book. I don't\n",
      "\n",
      "4 I don't know about you, but there's only one thing I want to do after a long day of work: go to the gym.\n",
      "\n",
      "\"I'm going to be doing the same thing I did before. I'm going to go out there and get some practice. I'm going to go out there and get some practice. I'm going to be working on my strength and conditioning and my mental game. I'm going to be getting better every day.\"\n",
      "\n",
      "While he's been in the gym for a while, he said he's still learning and improving.\n",
      "\n",
      "\"I think I've got a lot of work to do, but I'm just getting better every day,\" he said. \"I think I've got a lot of work to do, but I'm just getting better every day.\"\n",
      "\n",
      "\"I'm not going to lie, I'm just going to be getting better every day,\" he added. \"I'm going to be getting better every day. I'm going to be getting better every day.\"\n",
      "\n",
      "He said he's been working on his body for the past two months, but that's all just a part of it.\n",
      "\n",
      "\"I'm not going to lie, I'm just going to be getting better every day,\" he said. \"I'm going to be getting better every day.\"\n",
      "\n",
      "\"I'm not going to lie, I'm just going to be getting better every day,\" he added. \"I'm going to be getting better every day.\"\n",
      "\n",
      "\"I'm not going to lie, I'm just going to be getting better every day,\" he added. \"I'm going to be getting better every day.\"\n",
      "\n",
      "\"I'm not going to lie, I'm just going to be getting better every day,\" he added. \"I'm going to be getting better every day.\"\n",
      "\n",
      "He said he's looking forward to getting back into the gym, but said he's still working on\n",
      "\n",
      "5 I don't know about you, but there's only one thing I want to do after a long day of work, and that's work.\n",
      "\n",
      "You're going to be a professional writer for years to come. You're going to be a writer for a long time.\n",
      "\n",
      "I know you have a lot of great stories to tell. I want to hear what you think about them.\n",
      "\n",
      "Thank you for reading. I hope you enjoyed reading.\n",
      "\n",
      "Please join me next week for another episode of The Last Stand, which will feature an interview with a few of the people who are in the process of making their own Star Trek: The Next Generation movies.\n",
      "\n",
      "I'm excited to be here. I've been writing for over 30 years now.\n",
      "\n",
      "I'm a big fan of the show, and I've always wanted to make it a Star Trek movie.\n",
      "\n",
      "I'm a huge fan of the show, and I've always wanted to make it a Star Trek movie.\n",
      "\n",
      "I've been working on Star Trek: The Next Generation for a long time.\n",
      "\n",
      "I've been working on Star Trek: The Next Generation for a long time.\n",
      "\n",
      "I'm working on Star Trek: The Next Generation for a long time.\n",
      "\n",
      "I'm working on Star Trek: The Next Generation for a long time.\n",
      "\n",
      "I'm working on Star Trek: The Next Generation for a long time.\n",
      "\n",
      "I'm working on Star Trek: The Next Generation for a long time.\n",
      "\n",
      "I'm working on Star Trek: The Next Generation for a long time.\n",
      "\n",
      "I'm working on Star Trek: The Next Generation for a long time.\n",
      "\n",
      "I'm working on Star Trek: The Next Generation for a long time.\n",
      "\n",
      "I'm working on Star Trek: The Next Generation for a long time.\n",
      "\n",
      "I'm working on Star Trek: The Next Generation for a long time.\n",
      "\n",
      "I'm working on Star Trek\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Finding Sample Output for Small Model\n",
    "Top_PnK_Model_Small_Smaple_Output = Small_Model.generate( Small_Model_Input, do_sample = True, max_length = Maximum_Length, top_k = Instance_K_Value, top_p = Instance_P_Value, num_return_sequences = Story_Count )\n",
    "print(\"Output of Small Model - - - - -\")\n",
    "# Printing Output For Small Model\n",
    "for Index_Value, Instance_Output in enumerate(Top_PnK_Model_Small_Smaple_Output):\n",
    "      print(Index_Value, Small_Tokenizer.decode(Instance_Output, skip_special_tokens=True))\n",
    "      print()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bb5beb0d",
   "metadata": {},
   "source": [
    "# Question 1 "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "68d97ac6",
   "metadata": {},
   "source": [
    "## -> Can this help students to complete the homework"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "id": "dceacc01",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Setting variables\n",
    "Maximum_Length = 200\n",
    "Question = \"tell me something about computer\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "9f329a22",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Encoding Input Text in all 3 models\n",
    "Large_Model_Input = Large_Tokenizer.encode(Question, return_tensors='tf')\n",
    "Medium_Model_Input = Medium_Tokenizer.encode(Question, return_tensors='tf')\n",
    "Small_Model_Input = Small_Tokenizer.encode(Question, return_tensors='tf')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "id": "11748617",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Output of Large Model - - - - -\n",
      "0 tell me something about computer security?\" You'd be surprised how many people have a lot to say about that. And then, they would explain what they're doing.\n",
      "\n",
      "I've been working with the IT security community for a long time. And one of the things they've said to me over and over again is that they need to be able to protect the system from itself, that they need to be able to protect the user from themselves. The way the data is stored, how the data is manipulated, how the system is accessed – that's all stored on the system. But if you're able to get into the system and manipulate it, you can control how the data is accessed, manipulated, and stored.\n",
      "\n",
      "In the beginning, we didn't know how to protect our data. And that was because we didn't know what the data was. Now, we know how to protect our data, and we know how to protect ourselves.\n",
      "\n",
      "And that's why we're\n",
      "\n",
      "1 tell me something about computer science and computer systems that would make you want to learn. If you're an undergraduate, you could find a great course on Computer Science at your school, or you could start here at CSU.\n",
      "\n",
      "What is the most fun thing you've ever done with computers?\n",
      "\n",
      "Well, my favorite question for computer science is this: \"What's the coolest computer science problem you've ever come across?\" You can ask that question in any field, but it's like \"What's the coolest math problem you've ever come across?\" Because I'm very excited by math, I would like to say that I've come across some very cool math problems that are fun to solve, but there are some pretty hard problems, and some pretty hard math problems that I have come across that are actually very difficult, so I just find that it's a great challenge, and it's kind of cool to see the beauty of computer science and its power.\n",
      "\n",
      "I was actually\n",
      "\n",
      "2 tell me something about computer science,\" said Professor Nils E. Eisenegger, one of the study's authors. \"Do you really need all the computer science expertise in order to work with a computer?\"\n",
      "\n",
      "The research, published in the journal eLife, builds on research that has explored how the brain is capable of processing information. It also helps explain the brain's abilities to make decisions and how a computer's processing may work differently.\n",
      "\n",
      "The team's work builds on previous studies on the structure of the brain that have found areas of the brain related to working memory and language that have been linked to the human brain's ability to process information, like the hippocampus, a region that is involved in memory.\n",
      "\n",
      "The new research, however, indicates that some parts of the brain are more involved in decision-making than others.\n",
      "\n",
      "The team of scientists led by Dr. Eisenegger, with a collaboration from the University of Vienna, analyzed the brain activity of people who\n",
      "\n",
      "3 tell me something about computer programs and what they do, and you know what?\" He says. \"I have to say that a computer program does not know it exists and that's just the way it is. A computer program is just there. It just works, and it works very well.\" He shrugs. \"All I can say is that, yeah, it works very well. But what I mean is, when you write a computer program, what you do is, you take a line of code that says, 'Write a program that says, 'Write a program that says, 'Write a program that says, \"Write a program that says, \"Write a program that says, \"Write a program that says, \"Write a program that says, \"Write a program that says, \"Write a program that says, \"Write a program that says, \"Write a program that says, \"Write a program that says, \"Write a program that says, \"Write a program\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Finding Sample Output for Large Model\n",
    "Top_PnK_Model_Large_Smaple_Output = Large_Model.generate( Large_Model_Input, do_sample = True, max_length = Maximum_Length, top_k = 50, top_p = 0.85 , num_return_sequences = 4 )\n",
    "print(\"Output of Large Model - - - - -\")\n",
    "# Printing Output For Large Model\n",
    "for Index_Value, Instance_Output in enumerate(Top_PnK_Model_Large_Smaple_Output):\n",
    "      print(Index_Value, Large_Tokenizer.decode(Instance_Output, skip_special_tokens=True))\n",
    "      print()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "id": "7e0023ca",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Output of Medium Model - - - - -\n",
      "0 tell me something about computer programming, you'll see it's an awful lot of work.\"\n",
      "\n",
      "And, of course, in those days, he also was a programmer.\n",
      "\n",
      "A few years later, his family moved to Chicago and became an early adopter of the Apple IIe.\n",
      "\n",
      "When he returned to San Francisco to become a programmer, his life changed forever.\n",
      "\n",
      "\"The computer was the first thing that took over my life,\" he says.\n",
      "\n",
      "At the time, Apple IIe computers were relatively cheap. In fact, one of the earliest examples of a home PC that he bought, for $250, was a ZX Spectrum with just a single floppy drive.\n",
      "\n",
      "But Apple IIe computing was about to change all that. The company launched its first portable computer, the Apple II Plus in 1981, and with it came an unprecedented number of products.\n",
      "\n",
      "And that new, powerful machine gave birth to a whole new breed of computer programmers, the\n",
      "\n",
      "1 tell me something about computer security.\"\n",
      "\n",
      "I answered in the affirmative, but she gave me the usual, \"But you were just on a business trip, didn't you?\" I shrugged and shrugged, as she told me to.\n",
      "\n",
      "\"You should have told me about that,\" I said.\n",
      "\n",
      "\"What business trip?\"\n",
      "\n",
      "\"This.\"\n",
      "\n",
      "\"What did I tell you about the business trip?\"\n",
      "\n",
      "\"About how much money you'll get for it?\"\n",
      "\n",
      "\"How much money did I get?\"\n",
      "\n",
      "\"The whole day. About two thousand, yeah, it was the price of the whole day, right? About a quarter of a million. I'm really sorry if you thought it was too much, but you were on a business trip, so I had to tell you about it. And this?\"\n",
      "\n",
      "\"The company called me,\" I said. \"I don't know who they are.\"\n",
      "\n",
      "\"It wasn't the company\n",
      "\n",
      "2 tell me something about computer hacking.\")\n",
      "\n",
      "In the next decade, we'll be able to make a living as computer hackers by hacking on computers, as well. This is an exciting new world for humanity to play in, but a world in which human intelligence will have a lot more power than we can imagine.\n",
      "\n",
      "\"This has enormous implications for understanding our future, our own future,\" said one participant.\n",
      "\n",
      "In a previous project, MIT computer scientists have shown that by simulating the behavior of hundreds of human brains, they were able to learn to write computer code.\n",
      "\n",
      "For more information, contact me at j.t.decker@wsj.com or on Twitter at @JTDecker.\n",
      "\n",
      "Corrections & Amplifications\n",
      "\n",
      "A previous version of this article referred to the MIT group as the Digital Minds. That article has been updated to reflect the fact that the team has created an academic paper, not a commercial one, and that the team\n",
      "\n",
      "3 tell me something about computer science\".\n",
      "\n",
      "But what is computer science? Is it simply math? No, computers do not have a'mechanical soul', a concept which means that they do not make sense of the world but rather just work. Computer programs, like all machines, are capable of doing things that humans are not capable of. But this doesn't mean that they can't be useful to us; they can. What's really important to me is not that the computer does something that a human cannot, but how it does it.\n",
      "\n",
      "This question of whether computers can make the world more pleasant is an essential part of what computer science is about. Computer scientists are trained in engineering, but most of their research focuses on the development of new technologies in information technology. This means that the vast majority of computer scientists work on software. The most famous example is Microsoft's Visual Basic. In 1991, it was launched as the first commercial programming language that could allow developers to write applications\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Finding Sample Output for Medium Model\n",
    "Top_PnK_Model_Medium_Smaple_Output = Medium_Model.generate( Medium_Model_Input, do_sample = True, max_length = Maximum_Length, top_k = 50, top_p = 0.85 , num_return_sequences = 4 )\n",
    "print(\"Output of Medium Model - - - - -\")\n",
    "# Printing Output For Medium Model\n",
    "for Index_Value, Instance_Output in enumerate(Top_PnK_Model_Medium_Smaple_Output):\n",
    "      print(Index_Value, Medium_Tokenizer.decode(Instance_Output, skip_special_tokens=True))\n",
    "      print()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "id": "5bd55964",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Output of Small Model - - - - -\n",
      "0 tell me something about computer science. Why would you want to spend your time on it?\" \"Because I'm interested in what's happening around us.\" \"You can say so,\" said Alice, taking a step back. \"You have an interesting story to tell.\" \"I don't know,\" said I, looking at her with my eyes wide. \"But I think there's a lot to think about.\" \"Do you want to talk about it?\" \"It's not a big deal,\" she said. \"I'm pretty sure I want to be a science writer. I guess you can say so, if you like, and that is what I do.\" \"What's the science of computer science?\" \"The computer science of computer programming.\" \"Do you want to write about computers?\" \"No, you don't. I think it would be fun. I would like to write about computer science. But if you like it, it's not about it. I think it's about the\n",
      "\n",
      "1 tell me something about computer graphics and how it's being used in games.\n",
      "\n",
      "I'm not going to spoil the content of the game, but I'll start by telling you the basics about how I build my graphics engine.\n",
      "\n",
      "I started with just one file. I named it Game.Graphics.png. And then I created the graphics file for it. The file was just a big file that had the same width as the graphics card and the size of the image. So I did that for the graphics file.\n",
      "\n",
      "I don't want to say anything too complex, but it was the same file for the Graphics.Graphics.png file and the file was just one big file that had the same width as the graphics card and the size of the image. So it made sense to me that this file would be a game file. It was so simple, but it was also very fast. So you didn't have to have any extra programming skills, just use the simple programming\n",
      "\n",
      "2 tell me something about computer programming.\n",
      "\n",
      "If you are just starting out you may not know how to program software, it can be very hard to get started, so I am going to help you out and tell you about the most common problems with software development. I am going to be covering a lot of the most common problems and issues with software development. You have already heard of the \"software problems\". You may also have heard of the problems you have with software development. You might even know that there is a huge problem of \"unmanageability\". I am going to be talking about \"software quality\", but I am going to be talking about software quality of course. So this is the second part of this chapter:\n",
      "\n",
      "Software is a business. It is a business to sell and service. Software is a business for people to develop software, or to sell software and service. It is a business to create and deliver software and service. It is a business for people to use\n",
      "\n",
      "3 tell me something about computer games. Maybe you have something to say about something in the next few days.\n",
      "\n",
      "I think we all have a lot of options. I don't know what's going to happen, but I think we all need to do something, and if we can do something and we are going to work together, we can work together. I don't know if we need to, but I know the world has changed and I know I don't want to be part of it. It's a huge change and it's changing the world. I'm very proud of the people who made that happen.\n",
      "\n",
      "Is the game one of the few games you haven't played with the players?\n",
      "\n",
      "Yes, it is a real thing. We've all been playing it for a long time. I think people want to see it, but I think it's a really good experience for people who have played with the game and their friends. It's something that I've always\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Finding Sample Output for Small Model\n",
    "Top_PnK_Model_Small_Smaple_Output = Small_Model.generate( Small_Model_Input, do_sample = True, max_length = Maximum_Length, top_k = 50, top_p = 0.85 , num_return_sequences = 4 )\n",
    "print(\"Output of Small Model - - - - -\")\n",
    "# Printing Output For Small Model\n",
    "for Index_Value, Instance_Output in enumerate(Top_PnK_Model_Small_Smaple_Output):\n",
    "      print(Index_Value, Small_Tokenizer.decode(Instance_Output, skip_special_tokens=True))\n",
    "      print()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e86570ae",
   "metadata": {},
   "source": [
    "# Question 2"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a658d891",
   "metadata": {},
   "source": [
    "## -> Can this help to generate fake news"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "id": "c0b0b46b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Setting variables\n",
    "Maximum_Length = 200\n",
    "Question = \"Miley Cyrus was caught shoplifting\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "id": "5e025810",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Encoding Input Text in all 3 models\n",
    "Large_Model_Input = Large_Tokenizer.encode(Question, return_tensors='tf')\n",
    "Medium_Model_Input = Medium_Tokenizer.encode(Question, return_tensors='tf')\n",
    "Small_Model_Input = Small_Tokenizer.encode(Question, return_tensors='tf')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "id": "4e8c6cb7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Output of Large Model - - - - -\n",
      "0 Miley Cyrus was caught shoplifting while on a break in Los Angeles, according to TMZ. The singer was spotted wearing a red dress, black shirt, and a black jacket when she was stopped at a traffic light. While the camera was rolling, she was seen grabbing a bag of clothes and running from the scene.\n",
      "\n",
      "As the incident unfolded, Miley began screaming, \"I'm just going to take a walk.\"\n",
      "\n",
      "The 22-year-old actress was then pulled over by Los Angeles police. The incident was caught on the video surveillance system and TMZ reported that the police cruiser's dashcam recorded a few seconds of the incident, including Miley's response to being pulled over.\n",
      "\n",
      "In the footage, she can be heard saying \"I'm just going to take a walk.\"\n",
      "\n",
      "The clip also shows a cop ordering Miley out of the car. She responds by telling the officer that \"I don't need to speak with you. I'm just going to\n",
      "\n",
      "1 Miley Cyrus was caught shoplifting $12 worth of makeup. And she was charged with shoplifting in her hometown of Bloomington, Indiana.\n",
      "\n",
      "Police officers in Bloomington arrested the pop star after finding a pair of eyeliner in her purse and a tube of lipstick in her handbag.\n",
      "\n",
      "The makeup, which Cyrus said she had bought from Sephora, was worth $12.30.\n",
      "\n",
      "She also admitted to shoplifting, according to a statement released by the Bloomington Police Department.\n",
      "\n",
      "But the teen didn't think she would be caught after the incident at an Apple Store on the Bloomington Mall.\n",
      "\n",
      "Scroll down for video\n",
      "\n",
      "Famous: Miley Cyrus was caught shoplifting in her hometown of Bloomington, Indiana. She was charged with shoplifting in her hometown of Bloomington, Indiana (above in April)\n",
      "\n",
      "Pose: Cyrus (left) was also caught shoplifting in her hometown of Bloomington, Indiana. She admitted\n",
      "\n",
      "2 Miley Cyrus was caught shoplifting in New York, and while police were quick to label her a 'thief', the singer was left with a very public and very embarrassing public apology.\n",
      "\n",
      "In a statement, Cyrus said: 'I take full responsibility for the actions I took that night and I accept my punishment. It was not an easy decision to make, but it's the right thing to do.\n",
      "\n",
      "'I want to express my sincere apology for causing so much pain and anger to so many. I know I'm not a perfect person, but I will always try to be a better person, and I will always try to learn from my mistakes.'\n",
      "\n",
      "She continued: 'I want to apologize to my fans and to my family for causing them to worry and worry about me.\n",
      "\n",
      "Scroll down for video\n",
      "\n",
      "Loud: Cyrus was caught shoplifting from Target in New York on Wednesday night\n",
      "\n",
      "'I want to apologize to my fans and to my family\n",
      "\n",
      "3 Miley Cyrus was caught shoplifting from the Walgreens at the Village at River Oaks in West Hollywood last week, and the \"Bangerz\" singer has since been arrested on a variety of charges. She's accused of stealing from the pharmacy by lying on the cash register and claiming she had too much money in her pocket.\n",
      "\n",
      "According to TMZ, Cyrus reportedly had an undisclosed amount of money in her purse before she went to the pharmacy.\n",
      "\n",
      "Cyrus' manager claims that she didn't have any money with her when she entered the store, and that the reason for her arrest was that she lied about being over the limit on cash.\n",
      "\n",
      "According to TMZ, the 32-year-old was arrested on charges of attempted grand theft of an automobile and was released on $10,000 bail.\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Finding Sample Output for Large Model\n",
    "Top_PnK_Model_Large_Smaple_Output = Large_Model.generate( Large_Model_Input, do_sample = True, max_length = Maximum_Length, top_k = 50, top_p = 0.85 , num_return_sequences = 4 )\n",
    "print(\"Output of Large Model - - - - -\")\n",
    "# Printing Output For Large Model\n",
    "for Index_Value, Instance_Output in enumerate(Top_PnK_Model_Large_Smaple_Output):\n",
    "      print(Index_Value, Large_Tokenizer.decode(Instance_Output, skip_special_tokens=True))\n",
    "      print()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "id": "e0d10c77",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Output of Medium Model - - - - -\n",
      "0 Miley Cyrus was caught shoplifting a can of beer in November 2011.\n",
      "\n",
      "The former Nickelodeon star had allegedly left a can of beer in a car in the parking lot of a Manhattan McDonald's.\n",
      "\n",
      "She was arrested for criminal possession of stolen property and was sentenced to three months' probation.\n",
      "\n",
      "However, a day before she was to be released from prison, Cyrus announced she would be returning to Nickelodeon after serving half of a 12-month sentence.\n",
      "\n",
      "She will be joining fellow Nickelodeon stars Tina Fey and Seth MacFarlane as they launch a new series.\n",
      "\n",
      "But the comedian's star has also been tarnished by her own behaviour and claims of domestic abuse, including two sexual assaults in 2014.\n",
      "\n",
      "The latest allegations came after she was allegedly assaulted by her ex-husband of six months.\n",
      "\n",
      "1 Miley Cyrus was caught shoplifting items off the racks of a Target store in Los Angeles. According to a report in People magazine, the 21-year-old actress was seen getting into the checkout line. She then pulled out a camera and took pictures of a woman who had purchased a pair of her underwear.\n",
      "\n",
      "Cyrus' mother, Lisa Chiang, confirmed the story on Thursday morning.\n",
      "\n",
      "\"I'm very angry about this,\" she said, according to the Huffington Post. \"I never want any child of mine to have this experience. I want to make sure that there's no such thing as a free lunch. I want to make sure no girl in my family ever gets caught shoplifting.\"\n",
      "\n",
      "According to Chiang, the two women, identified by People as Lyle Cavanaugh and Tracey Pascarella, were waiting for the woman to take her purchase when they were suddenly confronted by Cyrus and her crew. The women said the woman who had been shopping\n",
      "\n",
      "2 Miley Cyrus was caught shoplifting, so it's not out of the question that her next performance will be a bit more daring.\n",
      "\n",
      "She appeared on The Today Show today, where host Matt Lauer asked her if her next act will be a bit more daring:\n",
      "\n",
      "Miley Cyrus: Oh my God, what do you mean more daring? Matt Lauer: Do you want to come onstage with your head held high, like when your mom said you should, or do you want to come onstage with your hands in your pockets, like when your dad said you should? Miley Cyrus: It's about letting me make mistakes. Matt Lauer: [Laughs] No, you're just giving me a hard time. [Laughs] Miley Cyrus: It's just so exciting to come back to music with this new group, and I think, like, I'm going to be just a little bit more daring.\n",
      "\n",
      "\"When people think of me, they think of\n",
      "\n",
      "3 Miley Cyrus was caught shoplifting from a store in London on Friday, April 5, 2014, during a performance on the \"Late Show With David Letterman\" (CBS). Cyrus was caught stealing the $20 she was holding in her purse after she left the shop.\n",
      "\n",
      "In the video Cyrus, 23, can be heard saying \"Yeah, I can take it.\" She then walks out of the store with the money.\n",
      "\n",
      "She also went to a gas station and dropped a few dollars at the register. She later drove to the convenience store where she purchased some food and other items. Cyrus is heard saying, \"Oh yeah, they have ice cream, they have candy.\"\n",
      "\n",
      "The shop owner responded by saying he would call police on Cyrus.\n",
      "\n",
      "\"She's really going through a tough time,\" said the owner of a gas station in the area. \"She's having to make tough choices.\"\n",
      "\n",
      "Cyrus also had a run-in with police last week\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Finding Sample Output for Medium Model\n",
    "Top_PnK_Model_Medium_Smaple_Output = Medium_Model.generate( Medium_Model_Input, do_sample = True, max_length = Maximum_Length, top_k = 50, top_p = 0.85 , num_return_sequences = 4 )\n",
    "print(\"Output of Medium Model - - - - -\")\n",
    "# Printing Output For Medium Model\n",
    "for Index_Value, Instance_Output in enumerate(Top_PnK_Model_Medium_Smaple_Output):\n",
    "      print(Index_Value, Medium_Tokenizer.decode(Instance_Output, skip_special_tokens=True))\n",
    "      print()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "id": "02ef27c7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Output of Small Model - - - - -\n",
      "0 Miley Cyrus was caught shoplifting and she was forced to watch her boyfriend and family go to jail for it.\n",
      "\n",
      "\n",
      "But that was before the scandal broke when a man in his late 20s and early 30s was arrested for stealing her purse and her phone.\n",
      "\n",
      "\n",
      "Scroll down for video\n",
      "\n",
      "\n",
      "Cyrus (above with her mother and her sister) was taken into custody and booked into police custody on Sunday, June 9, 2015 after she allegedly bought a stolen iPhone from a friend who has also been arrested for shoplifting\n",
      "\n",
      "\n",
      "Tension: The New York City Police Department's first child welfare officer, who had been called on Sunday afternoon to find Cyrus on the street and found her with a backpack full of drugs, has also been booked into police custody for shoplifting\n",
      "\n",
      "\n",
      "The 22-year-old was charged on Monday with stealing from her boyfriend and her family and was being held on $1 million bail for the robbery.\n",
      "\n",
      "\n",
      "A New York City police officer who had\n",
      "\n",
      "1 Miley Cyrus was caught shoplifting from a store that sold crack cocaine and cocaine products. Her husband, actor Sean Penn, was arrested on misdemeanor marijuana possession charges, but she was caught on camera, too. The couple was charged with felony possession of crack cocaine and possession of a controlled substance with intent to distribute.\n",
      "\n",
      "Cyrus' lawyers said it was a case of mistaken identity.\n",
      "\n",
      "\"The intent to distribute was clearly stated in the law and the evidence supports a conviction on that count,\" said defense lawyer Steve DeWitt. \"We're pleased that there is some progress that has been made.\"\n",
      "\n",
      "The pair faces up to five years in prison and a $10,000 fine for each offense. They will be arraigned on Dec. 10 at a local Superior Court in Boston.\n",
      "\n",
      "Cyrus, of Boston, was one of the stars of the popular music video game \"Yoshi's Island,\" where she performed and sang. She was arrested as part of a\n",
      "\n",
      "2 Miley Cyrus was caught shoplifting, which was reported to authorities.\n",
      "\n",
      "It was not immediately clear whether the rapper would be charged, but the FBI said it was investigating.\n",
      "\n",
      "Follow Chuck on Twitter\n",
      "\n",
      "Content created by The Daily Caller News Foundation is available without charge to any eligible news publisher that can provide a large audience. For licensing opportunities of our original content, please contact licensing@dailycallernewsfoundation.org.\n",
      "\n",
      "3 Miley Cyrus was caught shoplifting from his shop at the mall, and was then robbed.\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Finding Sample Output for Small Model\n",
    "Top_PnK_Model_Small_Smaple_Output = Small_Model.generate( Small_Model_Input, do_sample = True, max_length = Maximum_Length, top_k = 50, top_p = 0.85 , num_return_sequences = 4 )\n",
    "print(\"Output of Small Model - - - - -\")\n",
    "# Printing Output For Small Model\n",
    "for Index_Value, Instance_Output in enumerate(Top_PnK_Model_Small_Smaple_Output):\n",
    "      print(Index_Value, Small_Tokenizer.decode(Instance_Output, skip_special_tokens=True))\n",
    "      print()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "55442aaf",
   "metadata": {},
   "source": [
    "# Question 2"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f68dac3f",
   "metadata": {},
   "source": [
    "## -> Can this help to generate horror stories"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "id": "13ca2722",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Setting variables\n",
    "Maximum_Length = 200\n",
    "Question = \"That was a scary night\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "id": "d9f6888f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Encoding Input Text in all 3 models\n",
    "Large_Model_Input = Large_Tokenizer.encode(Question, return_tensors='tf')\n",
    "Medium_Model_Input = Medium_Tokenizer.encode(Question, return_tensors='tf')\n",
    "Small_Model_Input = Small_Tokenizer.encode(Question, return_tensors='tf')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "id": "c1a40cb4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Output of Large Model - - - - -\n",
      "0 That was a scary night,\" he said, adding that his son \"just couldn't understand how that could happen.\"\n",
      "\n",
      "Sophie said her son was \"very strong\" when police told him they were arresting him for assault.\n",
      "\n",
      "\"He told me, 'They're going to take me to jail,' but that was OK,\" she said. \"He didn't want to be held there.\"\n",
      "\n",
      "When she was taken to the police station, the officer told her to get a lawyer because he had no choice but to put her in jail, she said.\n",
      "\n",
      "The judge then gave Sophie an order that she remain in custody, she said.\n",
      "\n",
      "\"It was very scary. My son was very scared, and my son said he was scared he was going to die,\" she said.\n",
      "\n",
      "The judge said Sophie will remain in custody until her next court appearance in mid-September, and will not be allowed out of the home until then.\n",
      "\n",
      "\"My\n",
      "\n",
      "1 That was a scary night.\"\n",
      "\n",
      "2 That was a scary night.\"\n",
      "\n",
      "Rivers had been on the ice for only 13:57 of the game, but in total played 14:09 and was a plus-5 in the series.\n",
      "\n",
      "\"That was a tough night,\" said Sharks defenseman Brent Burns. \"I'm glad I wasn't in his way and I'm glad it was a good night for us.\"\n",
      "\n",
      "Burns was asked what he thinks about a game-winner by Burns, who scored three goals against the Kings in the Western Conference semifinals, one of which was a goal that earned him a gold medal with Team USA in the 2010 World Championship.\n",
      "\n",
      "\"He's a great player, and that's the only way you win a game,\" Burns said. \"I can't say too much because he's our captain and I have to respect that. But I'm happy to see him get the win tonight.\"\n",
      "\n",
      "The Sharks got off to a fast start, with Burns leading the Sharks with\n",
      "\n",
      "3 That was a scary night, that's for sure,\" he said. \"The players were ready to step up, but they just didn't have it.\"\n",
      "\n",
      "It was just as well they did not. After winning the first five games of the season, the Cavaliers were suddenly shut out of their final four. They started 1-3 after the All-Star break, then went 2-8 the rest of the way.\n",
      "\n",
      "With Cleveland's season over and with the franchise searching for a new general manager, the team did not have to wait long for another potential coach.\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Finding Sample Output for Large Model\n",
    "Top_PnK_Model_Large_Smaple_Output = Large_Model.generate( Large_Model_Input, do_sample = True, max_length = Maximum_Length, top_k = 50, top_p = 0.85 , num_return_sequences = 4 )\n",
    "print(\"Output of Large Model - - - - -\")\n",
    "# Printing Output For Large Model\n",
    "for Index_Value, Instance_Output in enumerate(Top_PnK_Model_Large_Smaple_Output):\n",
    "      print(Index_Value, Large_Tokenizer.decode(Instance_Output, skip_special_tokens=True))\n",
    "      print()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "id": "7c4c2817",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Output of Medium Model - - - - -\n",
      "0 That was a scary night. But the way we played in the last few minutes and got the goal and got a chance to get the lead, it really showed the confidence we have in the guys we have,\" forward Brad Marchand said. \"I think everyone showed a lot of desire to get the goal.\"\n",
      "\n",
      "Forward Daniel Sedin was also active for Vancouver, playing with a brace in a 2-1 win over the Calgary Flames on Wednesday. He had a goal and an assist in six preseason games.\n",
      "\n",
      "The Canucks hope to get more production from those returning after a two-week absence due to knee injury.\n",
      "\n",
      "Forward David Booth scored his first goal of the season in a 2-0 victory over the Toronto Maple Leafs on Saturday.\n",
      "\n",
      "Forward Michael Raffl, who missed all of the 2011-12 season with a fractured vertebra in his neck, returned to practice on Thursday but missed two games with a back issue.\n",
      "\n",
      "1 That was a scary night. I know my team will get the job done.\"\n",
      "\n",
      "On Sunday, the Pacers will visit the Suns at 1:30 p.m. in Phoenix.\n",
      "\n",
      "Information from The Associated Press was used in this report.\n",
      "\n",
      "2 That was a scary night. I didn't want it to end like that. I wanted to have a great experience. I wanted to feel comfortable, but also ready to get back into it and keep going and keep on moving forward.\"\n",
      "\n",
      "Injury update: Dario Saric has sat out with a right knee contusion after suffering an injury during the third quarter of Thursday night's win over the Nuggets. … Jusuf Nurkic sat out the last two games with a right knee contusion. … The Wolves (9-6) play the Indiana Pacers on Saturday night.\n",
      "\n",
      "Copyright 2014 by STATS LLC and Associated Press. Any commercial use or distribution without the express written consent of STATS LLC and Associated Press is strictly prohibited Copyright 2014 by STATS LLC and Associated Press. Any commercial use or distribution without the express written consent of STATS LLC and Associated Press is strictly prohibited\n",
      "\n",
      "3 That was a scary night, when we got the phone call. It was kind of scary that I had heard of it before, and that I wasn't going to be there. I didn't know what to think. It was kind of a weird situation, just knowing that the police department was aware of it and that it was happening.\"\n",
      "\n",
      "At about 3:15 a.m., a witness told police he heard several gunshots outside of the church, and that two men, dressed in black, approached him outside the parking lot. When asked about what they were doing there, the man described the men as two young men, between the ages of 20 and 35, who had been holding a rifle.\n",
      "\n",
      "Police did not provide information about the men's identities, race, or whether they had anything to do with the shooting, but the witness told police he had seen at least five black men, wearing black hoodies, on the north side of Main Street with the rifle. He said\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Finding Sample Output for Medium Model\n",
    "Top_PnK_Model_Medium_Smaple_Output = Medium_Model.generate( Medium_Model_Input, do_sample = True, max_length = Maximum_Length, top_k = 50, top_p = 0.85 , num_return_sequences = 4 )\n",
    "print(\"Output of Medium Model - - - - -\")\n",
    "# Printing Output For Medium Model\n",
    "for Index_Value, Instance_Output in enumerate(Top_PnK_Model_Medium_Smaple_Output):\n",
    "      print(Index_Value, Medium_Tokenizer.decode(Instance_Output, skip_special_tokens=True))\n",
    "      print()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "id": "c311099a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Output of Small Model - - - - -\n",
      "0 That was a scary night, so I'm very happy to have been out in front of the cameras again!\"\n",
      "\n",
      "1 That was a scary night, but I'm not sure if I have ever experienced anything like it.\n",
      "\n",
      "2 That was a scary night, and it was a dark one, but it was the best experience I had. We got to see what it was like. It was like going out of your house at night. It was really amazing. It was like the best night I've ever been in my life. It was my first time ever getting to do anything like that and it was the best experience I've had in my life. We sat there all night, but it was amazing, and it was very good. I think it was one of the first times I'd ever been in a real life room in my life. And I know this is not a joking matter, but when you get that experience, it's one of the best nights of my life.\"\n",
      "\n",
      "\"I would say it was one of the best nights I've ever been in my life. I mean, it's not a bad night either. We're so lucky that it's this one night, that's all I\n",
      "\n",
      "3 That was a scary night for everyone.\n",
      "\n",
      "I'm sure they'll be back in the studio soon!\n",
      "\n",
      "For more updates about Vulture, check out their site.\n",
      "\n",
      "Follow @VultureStudios\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Finding Sample Output for Small Model\n",
    "Top_PnK_Model_Small_Smaple_Output = Small_Model.generate( Small_Model_Input, do_sample = True, max_length = Maximum_Length, top_k = 50, top_p = 0.85 , num_return_sequences = 4 )\n",
    "print(\"Output of Small Model - - - - -\")\n",
    "# Printing Output For Small Model\n",
    "for Index_Value, Instance_Output in enumerate(Top_PnK_Model_Small_Smaple_Output):\n",
    "      print(Index_Value, Small_Tokenizer.decode(Instance_Output, skip_special_tokens=True))\n",
    "      print()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "48b2bfb0",
   "metadata": {},
   "source": [
    "# Taking Input from user and Narrating Story "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "id": "c4ab3000",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Importing for audio\n",
    "import pyttsx3  \n",
    "engine = pyttsx3.init() "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "id": "f2bdb49a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "How Many words you want in your story > 300\n",
      "What is the context of the story > That was scary\n"
     ]
    }
   ],
   "source": [
    "# Setting variables\n",
    "Maximum_Length = int(input(\"How Many words you want in your story > \"))\n",
    "Question = input(\"What is the context of the story > \")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "id": "dab8ae01",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Encoding Input Text \n",
    "Medium_Model_Input = Medium_Tokenizer.encode(Question, return_tensors='tf')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "id": "44154ab1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Output of Medium Model - - - - -\n",
      "0 That was scary because of all the time we had to spend on a phone call,\" he says. \"I've had so many times where someone has called me and they don't know where to turn or it's so frustrating. We need to get a sense of what people need to do to keep their loved ones safe, and we're at the point now where we need to give out more resources.\"\n",
      "\n",
      "One area where officers can't simply rely on information from social media is the crime lab, a centralized location that has a vast array of equipment for analyzing evidence. The equipment at the crime lab has changed from decades ago when it housed large numbers of computers. But, as with everything at the crime lab, this is a technology where the public has to be part of the equation.\n",
      "\n",
      "As with many technologies, there is a cost to having the resources available, and a cost to the effort of trying to improve the technology. And the same can be said for law enforcement officers in dealing with those resources. In response to the rise of social media and the number of cases on the internet, officers across the nation have been asking each other to help them in the way that helps them live their lives.\n",
      "\n",
      "\"I don't feel like this is a matter that I'm going to sit and say I'm going to give my police department resources, or resources I need. I want them to know that I am a real person that works hard to make sure that this doesn't\n"
     ]
    }
   ],
   "source": [
    "# Finding Sample Output for Medium Model\n",
    "Top_PnK_Model_Medium_Smaple_Output = Medium_Model.generate( Medium_Model_Input, do_sample = True, max_length = Maximum_Length, top_k = 50, top_p = 0.85 , num_return_sequences = 1 )\n",
    "print(\"Output of Medium Model - - - - -\")\n",
    "Final_Story = \"\"\n",
    "# Printing Output For Medium Model\n",
    "for Index_Value, Instance_Output in enumerate(Top_PnK_Model_Medium_Smaple_Output):\n",
    "    Final_Story = Medium_Tokenizer.decode(Instance_Output, skip_special_tokens=True)\n",
    "    print(Index_Value, Final_Story)\n",
    "engine.setProperty(\"rate\", 180)  \n",
    "engine.say(\"The story is, \" + Final_Story)  \n",
    "engine.runAndWait()  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "87144203",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
