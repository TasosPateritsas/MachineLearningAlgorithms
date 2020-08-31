function [newVocabulary, newTermEntropy, new_f] = GetNewVocabulary(r, Vocabulary, e, f)
    %Return the top r words in the Vocabulary (newVocabulary) 
    %    based on their entropy value
    %Also return 
    %   newTermEntropy: their corresponding Entropy value
    %   new_f: the new Term Document Matrix based on the
    %                newVocabulary

    % ADD Your Code Here

    [entr, idx] = sort(e(:));
    [~, entr_idx] = ind2sub(size(e), idx(1 : r));
    
    newTermEntropy = entr(1 : r, 1);
    new_f = f(:, entr_idx);
    newVocabulary = Vocabulary(entr_idx, :);
end